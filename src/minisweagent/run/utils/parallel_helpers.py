"""Parallel execution helpers -- thread-local logging, heterogeneous and GPU-pool runners.

Extracted from ParallelAgent to keep the agent class focused on orchestration
while execution details live here.
"""

from __future__ import annotations

import concurrent.futures
import json
import logging
import os
import queue as queue_mod
import re
import shutil
import subprocess
import sys
import threading
import time
import traceback
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from minisweagent.agents.default import TerminatingException
from minisweagent.debug_runtime import emit_debug_log
from minisweagent.run.task_file import create_worktree, create_worktree_with_patch

logger = logging.getLogger(__name__)

# ============================================================================
# Thread-local stdout/stderr redirection
# ============================================================================

_stdout_lock = threading.Lock()

_thread_log_file = threading.local()
_redirect_installed = False
_original_stdout = None
_original_stderr = None


def _get_thread_log():
    return getattr(_thread_log_file, "file", None)


class _ThreadLocalStream:
    """Writes to thread-local log file when set, else to original stream."""

    def __init__(self, original):
        self._original = original

    def write(self, s):
        f = _get_thread_log()
        if f is not None:
            try:
                f.write(s)
                f.flush()
            except Exception:
                pass
        else:
            self._original.write(s)
            self._original.flush()

    def flush(self):
        f = _get_thread_log()
        if f is not None:
            try:
                f.flush()
            except Exception:
                pass
        else:
            self._original.flush()

    def __getattr__(self, name):
        return getattr(self._original, name)


def _install_redirect_once():
    global _redirect_installed, _original_stdout, _original_stderr
    if not _redirect_installed:
        _original_stdout = sys.stdout
        _original_stderr = sys.stderr
        sys.stdout = _ThreadLocalStream(_original_stdout)
        sys.stderr = _ThreadLocalStream(_original_stderr)
        _install_logging_redirect()
        _redirect_installed = True


def _install_logging_redirect():
    """Route minisweagent logger to thread-local file when in sub-agent thread."""
    ms_logger = logging.getLogger("minisweagent")

    def filter_main_thread(record):
        return _get_thread_log() is None

    def filter_sub_agent_thread(record):
        return _get_thread_log() is not None

    class ThreadLocalFileHandler(logging.Handler):
        def emit(self, record):
            f = _get_thread_log()
            if f is not None:
                try:
                    f.write(self.format(record) + "\n")
                    f.flush()
                except Exception:
                    self.handleError(record)

    for h in list(ms_logger.handlers):
        h.addFilter(filter_main_thread)
    th = ThreadLocalFileHandler()
    th.setFormatter(logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s"))
    th.addFilter(filter_sub_agent_thread)
    ms_logger.addHandler(th)


@contextmanager
def redirect_output_to_file(log_file: Path):
    """Redirect this thread's stdout/stderr to the log file. Sub-agent output goes to its own file."""
    _install_redirect_once()
    f = open(log_file, "a", encoding="utf-8")
    prev = getattr(_thread_log_file, "file", None)
    _thread_log_file.file = f
    try:
        yield
    finally:
        _thread_log_file.file = prev
        f.close()


# ============================================================================
# Path / git helpers
# ============================================================================


def replace_paths(text: str, repo_path: Path, worktree_path: Path) -> str:
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


def bootstrap_git_repo(repo_path: Path, console=None) -> bool:
    """Bootstrap a minimal git repository for non-git directories.

    Creates .git, adds .gitignore to exclude build artifacts, and creates
    an initial commit. This allows unified git diff-based patch generation.

    Returns True if successful, False otherwise.
    """
    try:
        subprocess.run(
            ["git", "init", "-b", "geak-bootstrap"],
            cwd=repo_path,
            check=True,
            capture_output=True,
            text=True,
            timeout=30,
        )

        gitignore_path = repo_path / ".gitignore"
        gitignore_content = "\n".join(
            [
                "# GEAK auto-generated gitignore for build artifacts",
                "build/",
                "*/build/",
                ".rocprofv3/",
                "__pycache__/",
                "*.pyc",
                "*.o",
                "*.so",
                "*.a",
                "*.log",
                "*.dat",
                "optimization_logs/",
                "*/_logs/",
                "CMakeCache.txt",
                "CMakeFiles/",
                ".pytest_cache/",
                "*.egg-info/",
                ".geak_resolved/",
                "traj.json",
            ]
        )
        if gitignore_path.exists():
            existing = gitignore_path.read_text()
            if "# GEAK auto-generated" not in existing:
                gitignore_path.write_text(existing + "\n" + gitignore_content)
        else:
            gitignore_path.write_text(gitignore_content)

        subprocess.run(
            ["git", "add", "-A"],
            cwd=repo_path,
            check=True,
            capture_output=True,
            text=True,
            timeout=120,
        )
        subprocess.run(
            [
                "git",
                "-c",
                "user.name=geak-bootstrap",
                "-c",
                "user.email=geak@local",
                "commit",
                "-m",
                "GEAK bootstrap commit",
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
        logger.info("Git repo bootstrapped successfully at %s", repo_path)
        return True

    except subprocess.CalledProcessError as e:
        error_msg = e.stderr or e.stdout or str(e)
        if console:
            console.print(f"[bold red]Failed to bootstrap git repo: {error_msg}[/bold red]")
        logger.error("Failed to bootstrap git repo: %s", error_msg)
        return False
    except Exception as e:
        if console:
            console.print(f"[bold red]Failed to bootstrap git repo: {e}[/bold red]")
        logger.error("Failed to bootstrap git repo: %s", e)
        return False


def create_copy_workdir(src: Path, dst: Path) -> Path:
    """Create an isolated work directory by copying *src* (for non-git repos)."""
    if dst.exists():
        try:
            shutil.rmtree(dst)
        except Exception:
            pass
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(src, dst, symlinks=True)
    return dst


# ============================================================================
# Execution runners
# ============================================================================


def run_parallel_heterogeneous(
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
    labels = [s.label or s.agent_class.__name__ for s in agent_specs]
    if console:
        console.print(f"[bold green]Running {num_agents} heterogeneous agents: {labels}[/bold green]")
    logger.info("Running %d heterogeneous agents: %s", num_agents, labels)

    base_patch_dir = base_patch_dir.resolve()
    worktree_base = base_patch_dir / "worktrees"
    worktree_base.mkdir(parents=True, exist_ok=True)
    repo_path_resolved = repo_path.resolve()

    def run_spec_agent(agent_id: int, spec):
        """Run one agent from an AgentSpec."""
        if is_git_repo:
            worktree_path = create_worktree(repo_path, worktree_base / f"agent_{agent_id}")
        else:
            worktree_path = create_copy_workdir(repo_path, worktree_base / f"agent_{agent_id}")
            bootstrap_git_repo(worktree_path, console)
        worktree_path_str = str(worktree_path.resolve())

        label = spec.label or spec.agent_class.__name__
        if console:
            with _stdout_lock:
                console.print(
                    f"[bold green]Agent {agent_id} ({label}): "
                    f"GPU {spec.hip_visible_devices}, worktree {worktree_path}[/bold green]"
                )
        logger.info("Agent %d (%s): GPU %s, worktree %s", agent_id, label, spec.hip_visible_devices, worktree_path)

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
            parallel_agent_config["test_command"] = replace_paths(
                parallel_agent_config["test_command"], repo_path, worktree_path
            )

        task_with_repo = replace_paths(task_content, repo_path, worktree_path)

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
                    save_traj_fn(agent, parallel_output, exit_status=exit_status, result=result, extra_info=extra_info)

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
                logger.error("Error in heterogeneous agent %d: %s", agent_id, e, exc_info=True)
    return results


def run_pool(
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

    Unlike run_parallel_heterogeneous (which runs exactly N agents on N GPUs),
    this function accepts M tasks (where M can be > N) and schedules them across
    N GPU slots using a thread pool. When a task finishes and frees a GPU slot,
    the next queued task starts immediately -- like ProcessPoolExecutor.

    Args:
        tasks: List of AgentTask objects (from agent_spec.py), sorted by priority.
        gpu_ids: Available GPU device IDs (determines pool size N).
        base_task_content: Fallback task text if a task has no .task set.
        Other args: Same as run_parallel.
    """
    n_slots = len(gpu_ids)
    n_tasks = len(tasks)

    labels = [t.label or t.agent_class.__name__ for t in tasks]
    if console:
        console.print(
            f"[bold green]GPU Pool: {n_tasks} tasks on {n_slots} GPU slots "
            f"(labels: {labels[:8]}{'...' if len(labels) > 8 else ''})[/bold green]"
        )
    logger.info("GPU Pool: %d tasks on %d GPU slots (labels: %s)", n_tasks, n_slots, labels[:8])

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
            logger.info("Task %d (%s): assigned to GPU(s) %s (slot %d)", task_id, label, hip_devices, slot_idx)

            # Create or reset worktree for this slot
            wt_path = worktree_base / f"slot_{slot_idx}"
            if is_git_repo:
                starting_patch = task.config.get("starting_patch")
                if starting_patch:
                    create_worktree_with_patch(repo_path, wt_path, starting_patch)
                else:
                    create_worktree(repo_path, wt_path)
            else:
                create_copy_workdir(repo_path, wt_path)
                bootstrap_git_repo(wt_path, console)
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
                cfg["test_command"] = replace_paths(cfg["test_command"], repo_path, wt_path)

            # Resolve task text
            agent_task = task.task if task.task else base_task_content
            agent_task = replace_paths(agent_task, repo_path, wt_path)

            # Create model and environment with GPU assignment
            parallel_model = model_factory()
            base_env = env_factory()
            env_config_dict = base_env.config.__dict__.copy() if hasattr(base_env, "config") else {}
            env_config_dict["cwd"] = wt_path_str
            # Create a NEW dict to avoid shared-reference race across threads
            patched_env = {
                **(env_config_dict.get("env") or {}),
                "HIP_VISIBLE_DEVICES": hip_devices,
                "GEAK_WORK_DIR": wt_path_str,
                "GEAK_REPO_ROOT": str(repo_path.resolve()),
                "GEAK_GPU_DEVICE": hip_devices,
            }
            geak_harness = patched_env.get("GEAK_HARNESS")
            if isinstance(geak_harness, str) and geak_harness:
                patched_env["GEAK_HARNESS"] = replace_paths(geak_harness, repo_path, wt_path)
            env_config_dict["env"] = patched_env
            parallel_env = type(base_env)(**env_config_dict)

            parallel_output = None
            if output:
                parallel_output = output.parent / f"{output.stem}_task_{task_id}{output.suffix}"

            # region agent log
            emit_debug_log(
                "parallel_agent.py:execute_task:before_run",
                "Launching parallel optimization worker",
                {
                    "task_id": task_id,
                    "label": label,
                    "slot_idx": slot_idx,
                    "gpu_devices": hip_devices,
                    "step_limit": cfg.get("step_limit"),
                    "geak_harness": patched_env.get("GEAK_HARNESS"),
                    "worktree": wt_path_str,
                    "patch_dir": str(task_patch_dir),
                    "patch_dir_entries": sorted(p.name for p in task_patch_dir.iterdir())[:10],
                    "parallel_output": str(parallel_output) if parallel_output else None,
                },
                hypothesis_id="H5",
            )
            # endregion

            _wm_bm_path = cfg.pop("baseline_metrics", None)
            _wm_bb_path = cfg.pop("benchmark_baseline", None)

            agent = task.agent_class(parallel_model, parallel_env, **cfg)
            if hasattr(agent, "base_repo_path"):
                agent.base_repo_path = repo_path_resolved
            if hasattr(agent, "log_file"):
                agent.log_file = log_file

            try:
                from minisweagent.memory.integration import (  # pylint: disable=import-error,no-name-in-module
                    is_working_memory_enabled,
                )

                if is_working_memory_enabled():
                    from minisweagent.memory.working_memory import (  # pylint: disable=import-error,no-name-in-module
                        WorkingMemory,
                    )

                    _wm_notebook_dir = None
                    if _wm_bm_path:
                        try:
                            _wm_notebook_dir = str(Path(_wm_bm_path).resolve().parent / "_working_memory")
                        except Exception:
                            _wm_notebook_dir = None
                    # Extract kernel name from baseline_metrics path
                    _wm_kernel_cat = "unknown"
                    if _wm_bm_path:
                        _km = re.search(r"geak_eval_L\d+_(.+?)_\d{8}_\d{6}", _wm_bm_path)
                        if _km:
                            _wm_kernel_cat = _km.group(1)
                    _wm = WorkingMemory(
                        kernel_category=_wm_kernel_cat,
                        max_steps=cfg.get("step_limit", int(os.environ.get("GEAK_AGENT_STEP_LIMIT", "100"))),
                        notebook_dir=_wm_notebook_dir,
                        notebook_writer_id=f"{task.label or f'task_{task_id}'}-slot-{slot_idx}",
                    )
                    _wm.load_baseline_from_artifacts(
                        baseline_metrics_path=_wm_bm_path,
                        benchmark_baseline_path=_wm_bb_path,
                    )
                    _wm.sync_notebook_baseline()
                    # V2: Generate profiler diagnosis from baseline_metrics
                    if _wm_bm_path and Path(_wm_bm_path).exists():
                        try:
                            _bm2 = json.loads(Path(_wm_bm_path).read_text())
                            _top = _bm2.get("top_kernels", [])
                            if len(_top) > 3:
                                _target = _top[0] if _top else {}
                                _target_pct = _target.get("pct_of_total", 0)
                                _ext_pct = 100 - _target_pct
                                _top_summary = "; ".join(
                                    f"{k.get('name', '?')[:40]}: {k.get('duration_us', 0):.1f}us ({k.get('pct_of_total', 0):.0f}%)"
                                    for k in _top[:3]
                                )
                                if _ext_pct > 50:
                                    _wm.profiler_diagnosis = (
                                        f"[ARCHITECTURE ALERT] Profiler shows {len(_top)} sub-kernels. "
                                        f"Top 3: {_top_summary}. "
                                        f"No single kernel dominates (largest is {_target_pct:.0f}%). "
                                        "This usually means the entry point dispatches to UNFUSED external library calls. "
                                        "FIRST ACTION: Check triton_op() for try/except that falls through to aiter or other libraries. "
                                        "Bypass to use the local fused kernel. Also check for repeat_interleave or .contiguous() calls."
                                    )
                                elif _target_pct > 60:
                                    _wm.profiler_diagnosis = (
                                        f"[PROFILER] Target kernel ({_target.get('name', '?')[:40]}) dominates at {_target_pct:.0f}%. "
                                        "Focus optimization on the kernel body itself."
                                    )
                        except Exception:
                            pass
                    agent._working_memory = _wm
            except Exception:
                pass

            with open(log_file, "w", encoding="utf-8") as f:
                f.write(f"Task {task_id} ({label}) Conversation Log\n")
                f.write(f"GPU: {hip_devices} | Priority: {task.priority} | Language: {task.kernel_language}\n")
                f.write("=" * 60 + "\n\n")

            logger.info("[dim]Sub-agent %d (%s) started on GPU %s[/dim]", task_id, label, hip_devices)
            _agent_t0 = time.monotonic()
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
            _agent_elapsed = time.monotonic() - _agent_t0
            logger.info("Sub-agent %d (%s) finished in %.0fs (exit=%s)", task_id, label, _agent_elapsed, exit_status)

            # Auto-extract final patch from worktree if agent didn't save any
            if not list(task_patch_dir.glob("patch_*.patch")) and wt_path.exists():
                try:
                    _diff = subprocess.run(
                        ["git", "diff", "HEAD"],
                        cwd=str(wt_path),
                        capture_output=True,
                        text=True,
                        timeout=30,
                    )
                    if _diff.returncode == 0 and _diff.stdout.strip():
                        (task_patch_dir / "patch_0.patch").write_text(_diff.stdout)
                except Exception:
                    pass

            # region agent log
            emit_debug_log(
                "parallel_agent.py:execute_task:after_run",
                "Parallel optimization worker returned from agent.run",
                {
                    "task_id": task_id,
                    "label": label,
                    "slot_idx": slot_idx,
                    "gpu_devices": hip_devices,
                    "exit_status": str(exit_status),
                    "result_preview": (str(result)[:300] if result is not None else None),
                    "has_traceback": bool(extra_info and extra_info.get("traceback")),
                    "patch_count": len(list(task_patch_dir.glob("*.patch"))),
                    "best_results_present": (task_patch_dir / "best_results.json").exists(),
                },
                hypothesis_id="H7",
            )
            # endregion

            if console:
                with _stdout_lock:
                    console.print(f"[bold blue]Task {task_id} ({label}): completed on GPU(s) {hip_devices}[/bold blue]")
            logger.info("Task %d (%s): completed on GPU(s) %s", task_id, label, hip_devices)

            return task_id, agent, exit_status, result

        finally:
            for g in acquired_gpus:
                gpu_queue.put(g)

    # Progress reporting thread
    _progress_stop = threading.Event()
    _dispatch_t0 = time.monotonic()

    def _report_progress():
        _interval = float(os.environ.get("GEAK_PROGRESS_INTERVAL", "30"))
        while not _progress_stop.wait(_interval):
            elapsed = time.monotonic() - _dispatch_t0
            patches_by_task = []
            for _tid, _task in sorted_tasks:
                _lbl = _task.label or f"task_{_tid}"
                _pdir = base_patch_dir / _lbl
                count = len(list(_pdir.glob("*.patch"))) if _pdir.is_dir() else 0
                patches_by_task.append((_lbl, count))
            total = sum(c for _, c in patches_by_task)
            summary = ", ".join(f"{l}: {c}" for l, c in patches_by_task if c > 0)
            logger.info(
                "[dim][%.1fmin] Sub-agents working: %d total patches%s[/dim]",
                elapsed / 60,
                total,
                f" ({summary})" if summary else "",
                extra={"progress_tick": True},
            )

    _progress_thread = threading.Thread(target=_report_progress, daemon=True)
    _progress_thread.start()

    # Submit ALL M tasks; ThreadPoolExecutor(max_workers=N) queues overflow
    results = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=n_slots) as executor:
        futures = {executor.submit(execute_task, tid, task): tid for tid, task in sorted_tasks}
        # region agent log
        emit_debug_log(
            "parallel_agent.py:_run_pool:futures_submitted",
            "Submitted pool tasks to ThreadPoolExecutor",
            {
                "n_slots": n_slots,
                "n_tasks": n_tasks,
                "task_ids": [tid for tid, _task in sorted_tasks],
            },
            hypothesis_id="H8",
        )
        # endregion
        for future in concurrent.futures.as_completed(futures):
            try:
                r = future.result()
                results.append(r)
                # region agent log
                emit_debug_log(
                    "parallel_agent.py:_run_pool:future_completed",
                    "Pool future completed successfully",
                    {
                        "task_id": futures[future],
                        "results_collected": len(results),
                        "exit_status": str(r[2]) if len(r) > 2 else None,
                    },
                    hypothesis_id="H8",
                )
                # endregion
            except Exception as e:
                task_id = futures[future]
                logger.error("Error in pool task %d: %s", task_id, e, exc_info=True)
                # region agent log
                emit_debug_log(
                    "parallel_agent.py:_run_pool:future_exception",
                    "Pool future raised exception while collecting result",
                    {
                        "task_id": task_id,
                        "error_type": type(e).__name__,
                        "error": str(e),
                        "results_collected": len(results),
                    },
                    hypothesis_id="H8",
                )
                # endregion

    _progress_stop.set()
    _progress_thread.join(timeout=2)

    # region agent log
    emit_debug_log(
        "parallel_agent.py:_run_pool:after_all_futures",
        "All pool futures drained and _run_pool is returning",
        {
            "results_count": len(results),
            "task_ids_completed": sorted(
                int(r[0]) for r in results if isinstance(r, tuple) and len(r) > 0 and isinstance(r[0], int)
            ),
        },
        hypothesis_id="H9",
    )
    # endregion

    return results
