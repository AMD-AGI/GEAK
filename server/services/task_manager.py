"""Task management service for GEAK Online Service."""

import asyncio
import logging
import os
import signal
import subprocess
import threading
import uuid
import yaml
import shutil
from pathlib import Path
from datetime import datetime
from typing import Any

from server.config import get_settings
from server.database import TaskDB, UserConfigDB
from server.services.safe_client import SaFEClient
from server.api.schemas.task import TaskCreate, RepoInput

logger = logging.getLogger(__name__)


def _detect_available_gpus() -> list[int]:
    """Detect available AMD GPU device IDs via rocm-smi.

    Returns a sorted list of integer device IDs, or [0] as fallback.
    """
    try:
        result = subprocess.run(
            ["rocm-smi", "--showid", "--csv"],
            capture_output=True, text=True, timeout=10,
        )
        if result.returncode != 0:
            return [0]
        gpu_ids = []
        for line in result.stdout.strip().splitlines()[1:]:
            parts = line.split(",")
            if parts:
                try:
                    gpu_ids.append(int(parts[0].strip().replace("card", "")))
                except ValueError:
                    continue
        return sorted(gpu_ids) if gpu_ids else [0]
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return [0]


class TaskManager:
    """Service for managing optimization tasks."""

    _gpu_counter: int = 0
    _gpu_lock = threading.Lock()
    _available_gpus: list[int] | None = None
    
    def __init__(self, user_id: str, api_key: str):
        self.user_id = user_id
        self.api_key = api_key
        self.settings = get_settings()
        self.local_mode = os.getenv("GEAK_LOCAL", "false").lower() == "true"
        self._running_tasks: dict[str, int] = {}
        if not self.local_mode:
            self.safe_client = SaFEClient(api_key)
        else:
            self.safe_client = None
            if TaskManager._available_gpus is None:
                TaskManager._available_gpus = _detect_available_gpus()
                logger.info("Local mode: detected GPUs %s", TaskManager._available_gpus)

    def _allocate_gpu(self) -> int:
        """Allocate a GPU ID using round-robin across available GPUs."""
        gpus = TaskManager._available_gpus or [0]
        with TaskManager._gpu_lock:
            gpu_id = gpus[TaskManager._gpu_counter % len(gpus)]
            TaskManager._gpu_counter += 1
        logger.info("Allocated GPU %d for task (counter=%d, pool=%s)",
                     gpu_id, TaskManager._gpu_counter, gpus)
        return gpu_id
    
    def _get_task_dir(self, task_id: str) -> Path:
        """Get task directory path."""
        return Path(self.settings.nfs_base_path) / "tasks" / self.user_id / task_id
    
    def _get_input_dir(self, task_id: str) -> Path:
        """Get input directory for a task."""
        return self._get_task_dir(task_id) / "input"
    
    def _get_output_dir(self, task_id: str) -> Path:
        """Get output directory for a task."""
        return self._get_task_dir(task_id) / "output"
    
    def _get_default_prompt(self) -> str:
        """Get default prompt template."""
        template_path = Path(__file__).parent.parent / "templates" / "default_prompt.md"
        if template_path.exists():
            return template_path.read_text()
        return """# HIP Kernel Optimization Task

Please analyze and optimize the provided HIP kernel code for better performance on AMD GPUs.

Focus on:
1. Memory access patterns and coalescing
2. Occupancy optimization
3. Register usage
4. Shared memory utilization
5. Warp-level optimizations

Provide the optimized code with comments explaining the changes made.
"""
    
    def _get_default_config(self) -> dict:
        """Get default agent configuration from template file."""
        template_path = Path(__file__).parent.parent / "templates" / "default_config.yaml"
        
        if template_path.exists():
            with open(template_path, 'r') as f:
                config = yaml.safe_load(f)
            return config
        
        # Fallback minimal config if template not found
        return {
            "model": {
                "model_class": "amd_llm",
                "model_name": "gpt-5.2",
                "model_kwargs": {
                    "temperature": 0.0,
                    "max_tokens": 16000,
                },
            },
            "agent": {
                "step_limit": 50,
                "cost_limit": 10.0,
            },
        }
    
    def _build_env_vars(self, task_id: str) -> dict[str, str]:
        """Build environment variables for workload execution."""
        env_vars = {
            "MSWEA_CONFIGURED": "true",
            "TASK_ID": task_id,
        }
        
        # Langfuse tracing (zero-intrusion integration via litellm callbacks)
        if self.settings.langfuse_enabled and self.settings.langfuse_public_key:
            env_vars.update({
                "LITELLM_CALLBACKS": "langfuse",
                "LANGFUSE_PUBLIC_KEY": self.settings.langfuse_public_key,
                "LANGFUSE_SECRET_KEY": self.settings.langfuse_secret_key or "",
                "LANGFUSE_HOST": self.settings.langfuse_base_url,
                # Use task_id as session_id for easy tracing
                "LANGFUSE_SESSION_ID": task_id,
            })
        
        return {k: v for k, v in env_vars.items() if v is not None}
    
    async def _merge_config(self, user_config: dict | None) -> dict:
        """Merge user config with defaults.
        
        Priority (highest to lowest):
        1. User config passed in task creation request
        2. User's saved default model config (from database)
        3. System default config (from default_config.yaml)
        """
        # Deep merge helper
        def merge(base: dict, override: dict) -> dict:
            result = base.copy()
            for key, value in override.items():
                if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                    result[key] = merge(result[key], value)
                else:
                    result[key] = value
            return result
        
        # Start with system default
        config = self._get_default_config()
        
        # Apply user's saved default model config if exists
        user_saved_config = await UserConfigDB.get(self.user_id)
        if user_saved_config and user_saved_config.get("model_config"):
            saved_model_config = user_saved_config["model_config"]
            # Merge saved model config into the config's model section
            if "model" not in config:
                config["model"] = {}
            config["model"] = merge(config.get("model", {}), saved_model_config)
        
        # Apply user config from request (highest priority)
        if user_config:
            config = merge(config, user_config)
        
        # Normalize api_key placement based on model_class:
        # - amd_llm: expects api_key at model top level (AmdLlmModelConfig.api_key)
        # - litellm: expects api_key inside model_kwargs only (LitellmModelConfig rejects unknown fields)
        model_cfg = config.get("model", {})
        model_kwargs = model_cfg.get("model_kwargs", {})
        model_class = model_cfg.get("model_class", "")
        if model_class == "amd_llm":
            if "api_key" in model_kwargs and "api_key" not in model_cfg:
                model_cfg["api_key"] = model_kwargs["api_key"]
        else:
            if "api_key" in model_cfg:
                model_kwargs.setdefault("api_key", model_cfg.pop("api_key"))
                model_cfg["model_kwargs"] = model_kwargs
        
        return config
    
    def _get_runtime_config(self, user_runtime: dict | None) -> dict:
        """Get runtime config with defaults."""
        defaults = {
            "image": self.settings.default_image,
            "gpu_count": self.settings.default_gpu_count,
            "cpu": self.settings.default_cpu,
            "memory": self.settings.default_memory,
            "timeout": self.settings.default_timeout,
        }
        if user_runtime:
            for key, value in user_runtime.items():
                if value is not None:
                    defaults[key] = value
        return defaults
    
    async def create_task(self, request: TaskCreate) -> dict:
        """Create a new optimization task.
        
        Args:
            request: Task creation request.
        
        Returns:
            Created task information.
        """
        task_id = str(uuid.uuid4())
        task_dir = self._get_task_dir(task_id)
        input_dir = self._get_input_dir(task_id)
        output_dir = self._get_output_dir(task_id)
        
        # Create directories
        input_dir.mkdir(parents=True, exist_ok=True)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Handle input based on type
        if request.input_type == "file" and request.files:
            await self._save_files_input(input_dir, request.files)
            # Use the first file as primary input path (usually the main .hip file)
            input_path = str(input_dir / request.files[0].filename)
        elif request.input_type == "repo" and request.repo:
            input_path = await self._clone_repo(input_dir, request.repo)
        else:
            raise ValueError(f"Invalid input: {request.input_type} requires corresponding data")
        
        # Get prompt
        prompt = request.prompt or self._get_default_prompt()
        
        # Merge configurations (uses user's saved default if available)
        user_config = request.config.model_dump(exclude_none=True) if request.config else None
        config = await self._merge_config(user_config)
        
        user_runtime = request.runtime.model_dump(exclude_none=True) if request.runtime else None
        runtime_config = self._get_runtime_config(user_runtime)
        
        if request.workspace_id:
            runtime_config["workspace_id"] = request.workspace_id
        
        # Save prompt and config to task directory
        (task_dir / "prompt.md").write_text(prompt)
        (task_dir / "config.yaml").write_text(yaml.dump(config, default_flow_style=False))
        
        # Create task in database
        task = await TaskDB.create(
            task_id=task_id,
            user_id=self.user_id,
            input_type=request.input_type,
            input_path=input_path,
            prompt=prompt,
            config=config,
            runtime_config=runtime_config,
        )
        
        return task
    
    async def _save_files_input(self, input_dir: Path, files: list):
        """Save multiple file inputs.
        
        Args:
            input_dir: Directory to save files.
            files: List of FileInput objects with filename and content.
        """
        for file in files:
            file_path = input_dir / file.filename
            # Create subdirectories if filename contains path
            file_path.parent.mkdir(parents=True, exist_ok=True)
            file_path.write_text(file.content)
    
    async def _clone_repo(self, input_dir: Path, repo: RepoInput) -> str:
        """Clone git repository."""
        import subprocess
        
        repo_dir = input_dir / "repo"
        cmd = ["git", "clone"]
        if repo.branch:
            cmd.extend(["-b", repo.branch])
        cmd.extend(["--depth", "1", repo.url, str(repo_dir)])
        
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            raise Exception(f"Failed to clone repo: {result.stderr}")
        
        if repo.subdir:
            return str(repo_dir / repo.subdir)
        return str(repo_dir)
    
    async def submit_task(self, task_id: str) -> dict:
        """Submit task for execution (locally or via SaFE platform).
        
        Args:
            task_id: Task ID.
        
        Returns:
            Updated task with workload info.
        """
        if self.local_mode:
            return await self._submit_local(task_id)
        return await self._submit_remote(task_id)
    
    async def _submit_remote(self, task_id: str) -> dict:
        """Submit task to SaFE platform for execution."""
        task = await TaskDB.get(task_id)
        if not task:
            raise ValueError(f"Task not found: {task_id}")
        if task["user_id"] != self.user_id:
            raise PermissionError("Access denied")
        
        task_dir = self._get_task_dir(task_id)
        output_dir = self._get_output_dir(task_id)
        runtime = task.get("runtime_config", {})
        input_type = task.get("input_type", "file")
        
        # Build execution command based on input type
        input_path = task.get("input_path", "")
        command = self._build_execution_command(task_id, task_dir, output_dir, input_type, input_path)
        
        # Get workspace ID (from task config or default to user's first workspace)
        workspace_id = runtime.get("workspace_id") or await self.safe_client.get_default_workspace_id()
        
        # Create workload on SaFE
        workload = await self.safe_client.create_workload(
            workspace_id=workspace_id,
            name=f"geak-{task_id[:8]}",
            image=runtime.get("image", self.settings.default_image),
            command=command,
            gpu_count=runtime.get("gpu_count", self.settings.default_gpu_count),
            cpu=runtime.get("cpu", self.settings.default_cpu),
            memory=runtime.get("memory", self.settings.default_memory),
            env_vars=self._build_env_vars(task_id),
            volumes=[
                {
                    "name": "nfs-storage",
                    "mountPath": self.settings.nfs_base_path,
                    "nfs": {
                        "server": "nfs-server",
                        "path": self.settings.nfs_base_path,
                    },
                }
            ],
        )
        
        # Update task with workload ID
        workload_id = workload.get("id") or workload.get("workloadId")
        task = await TaskDB.update(
            task_id,
            status="running",
            safe_workload_id=workload_id,
            output_path=str(output_dir),
        )
        
        return task
    
    async def _submit_local(self, task_id: str) -> dict:
        """Run GEAK agent locally via subprocess.

        Each task is assigned a dedicated GPU via round-robin to avoid
        contention when multiple tasks run concurrently.
        """
        task = await TaskDB.get(task_id)
        if not task:
            raise ValueError(f"Task not found: {task_id}")
        if task["user_id"] != self.user_id:
            raise PermissionError("Access denied")
        
        task_dir = self._get_task_dir(task_id)
        output_dir = self._get_output_dir(task_id)
        log_path = output_dir / "execution.log"

        # Append fallback kernel path hint to prompt (same as remote mode)
        input_path = task.get("input_path", "")
        if input_path and task.get("input_type") == "file":
            input_dir = self._get_input_dir(task_id)
            with open(task_dir / "prompt.md", "a") as f:
                f.write(
                    f"\n\nNote: If the kernel path or kernel repo path above cannot be found, "
                    f"use kernel absolute path: {input_path} "
                    f"and kernel repo path: {input_dir}/ as the fallback.\n"
                )

        gpu_id = self._allocate_gpu()
        
        geak_bin = shutil.which("geak") or shutil.which("geak-gaagent") or "geak"
        cmd = [
            geak_bin,
            "-c", str(task_dir / "config.yaml"),
            "-t", str(task_dir / "prompt.md"),
            "-o", str(output_dir) + "/",
            "--gpu-ids", str(gpu_id),
            "--yolo",
        ]
        logger.info("Local task %s → GPU %d, cmd: %s", task_id[:8], gpu_id, " ".join(cmd))
        
        env = os.environ.copy()
        env.update(self._build_env_vars(task_id))
        env["HIP_VISIBLE_DEVICES"] = str(gpu_id)
        
        log_f = open(log_path, "w")
        try:
            proc = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=log_f,
                stderr=asyncio.subprocess.STDOUT,
                cwd=str(output_dir),
                env=env,
            )
        except FileNotFoundError:
            # geak CLI not found — mark as failed immediately
            log_f.write("ERROR: 'geak' command not found. Please install the GEAK agent.\n")
            log_f.close()
            return await TaskDB.update(
                task_id,
                status="failed",
                error_message="geak CLI not found in PATH",
                output_path=str(output_dir),
            )
        except Exception as e:
            log_f.write(f"ERROR: Failed to start subprocess: {e}\n")
            log_f.close()
            return await TaskDB.update(
                task_id,
                status="failed",
                error_message=str(e),
                output_path=str(output_dir),
            )

        self._running_tasks[task_id] = proc.pid
        task = await TaskDB.update(
            task_id,
            status="running",
            output_path=str(output_dir),
        )

        asyncio.create_task(self._wait_local_task(task_id, proc, log_f))
        return task
    
    async def _wait_local_task(self, task_id: str, proc, log_file):
        """Wait for local subprocess to complete and update status."""
        try:
            returncode = await proc.wait()
            status = "completed" if returncode == 0 else "failed"
            error_msg = None
            if returncode != 0:
                error_msg = f"Process exited with code {returncode}"
            await TaskDB.update(task_id, status=status, error_message=error_msg)
        except Exception as e:
            await TaskDB.update(task_id, status="failed", error_message=str(e))
        finally:
            self._running_tasks.pop(task_id, None)
            try:
                log_file.close()
            except Exception:
                pass
    
    def _build_execution_command(self, task_id: str, task_dir: Path, output_dir: Path, input_type: str = "file", input_path: str = "") -> str:
        """Build the execution command for the workload.
        
        Args:
            task_id: Task ID.
            task_dir: Task directory path.
            output_dir: Output directory path.
            input_type: Input type ('file' or 'repo').
            input_path: Absolute path to primary input file.
        """
        settings = self.settings
        
        # Build env export block from _build_env_vars
        env_vars = self._build_env_vars(task_id)
        env_export_lines = "\n".join(f'export {k}="{v}"' for k, v in env_vars.items())
        
        # Optional pre-command (e.g. trust certificates)
        precommand_block = ""
        if settings.entrypoint_precommand:
            precommand_block = f"""
# Pre-command (from ENTRYPOINT_PRECOMMAND)
{settings.entrypoint_precommand}
"""
        
        # Common setup
        setup_commands = f"""#!/bin/bash
set -e

# Export environment variables
{env_export_lines}

# Clone GEAK repository
cd /tmp
git clone -b {settings.geak_branch} {settings.geak_repo_url} geak
cd geak

# Install dependencies
pip install -e .

# Install langfuse for LLM tracing (v2.x compatible with litellm)
pip install 'langfuse>=2.0.0,<3.0.0' -q 2>/dev/null || true

git clone https://github.com/AMDResearch/intellikit.git
pip install -e  intellikit/metrix/
{precommand_block}
# Append fallback kernel path hint to prompt
echo "" >> "{task_dir}/prompt.md"
echo "Note: If the kernel path or kernel repo path above cannot be found, use kernel absolute path: {input_path} and kernel repo path: {task_dir}/input/ as the fallback." >> "{task_dir}/prompt.md"

# Set up task
TASK_DIR="{task_dir}"
OUTPUT_DIR="{output_dir}"
cd "$OUTPUT_DIR"
"""
        
        if input_type == "repo":
            # For repo input, run geak from the repo directory
            run_commands = f"""
# Set REPO_DIR environment variable for use in prompts
export REPO_DIR="$TASK_DIR/input/repo"

# Run optimization from repo directory
geak -c "$TASK_DIR/config.yaml" -t "$TASK_DIR/prompt.md" -o "$OUTPUT_DIR/" --yolo > "$OUTPUT_DIR/execution.log" 2>&1

# Archive the modified repo
cd "$TASK_DIR/input"
tar -czf "$OUTPUT_DIR/modified_repo.tar.gz" repo/

"""
        else:
            # For file input, run geak
            run_commands = f"""
# Run optimization
geak -c "$TASK_DIR/config.yaml" -t "$TASK_DIR/prompt.md" -o "$OUTPUT_DIR/" --yolo > "$OUTPUT_DIR/execution.log" 2>&1

"""
        
        finish_commands = f"""
echo "Task {task_id} completed successfully" >> "$OUTPUT_DIR/execution.log"
"""
        
        return setup_commands + run_commands + finish_commands
    
    async def get_task(self, task_id: str) -> dict | None:
        """Get task by ID.
        
        Also checks and updates task status if it's running.
        """
        task = await TaskDB.get(task_id)
        if task and task["user_id"] != self.user_id:
            return None
        
        if task:
            # Auto-detect completion status for running tasks
            task = await self._check_and_update_status(task)
            return self._enrich_task(task)
        return None
    
    async def _check_and_update_status(self, task: dict) -> dict:
        """Check and update task status by querying SaFE workload.
        
        This detects if a running task has completed or failed.
        """
        if task.get("status") != "running":
            return task
        
        if self.local_mode:
            if task["id"] not in self._running_tasks:
                return await self._check_log_for_completion(task)
            return task
        
        workload_id = task.get("safe_workload_id")
        if not workload_id:
            return task
        
        try:
            # Query SaFE for workload status
            workload = await self.safe_client.get_workload(workload_id)
            
            # Map SaFE status to task status
            safe_status = workload.get("status", "").lower()
            phase = workload.get("phase", "").lower()
            
            new_status = self._map_workload_status(safe_status, phase)
            
            if new_status and new_status != task["status"]:
                task = await TaskDB.update(task["id"], status=new_status)
        
        except Exception:
            # Fallback to log-based detection if SaFE query fails
            task = await self._check_log_for_completion(task)
        
        return task
    
    def _map_workload_status(self, status: str, phase: str) -> str | None:
        """Map SaFE workload status to task status."""
        # Check phase first (more specific)
        if phase in ("succeeded", "completed"):
            return "completed"
        elif phase == "failed":
            return "failed"
        elif phase in ("pending", "creating", "running"):
            return "running"
        
        # Check status
        if status == "succeeded":
            return "completed"
        elif status == "failed":
            return "failed"
        elif status in ("pending", "running"):
            return "running"
        
        return None
    
    async def _check_log_for_completion(self, task: dict) -> dict:
        """Fallback: Check execution log for completion markers."""
        output_dir = self._get_output_dir(task["id"])
        log_path = output_dir / "execution.log"
        
        if not log_path.exists():
            return task
        
        try:
            content = log_path.read_text()
            
            if "completed successfully" in content:
                task = await TaskDB.update(task["id"], status="completed")
        except Exception:
            pass
        
        return task
    
    async def list_tasks(
        self,
        status: str | None = None,
        limit: int = 20,
        offset: int = 0,
    ) -> tuple[list[dict], int]:
        """List tasks for current user."""
        tasks = await TaskDB.list_by_user(
            self.user_id,
            status=status,
            limit=limit,
            offset=offset,
        )
        # Get total count (simplified - could add count query)
        total = len(tasks)
        return [self._enrich_task(t) for t in tasks], total
    
    async def cancel_task(self, task_id: str) -> dict:
        """Cancel a running task."""
        task = await TaskDB.get(task_id)
        if not task:
            raise ValueError(f"Task not found: {task_id}")
        if task["user_id"] != self.user_id:
            raise PermissionError("Access denied")
        
        # Local mode: kill the subprocess
        if self.local_mode and task.get("status") == "running":
            pid = self._running_tasks.get(task_id)
            if pid:
                try:
                    os.kill(pid, signal.SIGTERM)
                except ProcessLookupError:
                    pass
                self._running_tasks.pop(task_id, None)
        
        # Remote mode: stop workload on SaFE if running
        if not self.local_mode and task.get("safe_workload_id") and task.get("status") == "running":
            await self.safe_client.stop_workload(task["safe_workload_id"])
        
        return await TaskDB.update(task_id, status="cancelled")
    
    async def get_outputs(self, task_id: str) -> dict:
        """Get task outputs information."""
        task = await TaskDB.get(task_id)
        if not task:
            raise ValueError(f"Task not found: {task_id}")
        if task["user_id"] != self.user_id:
            raise PermissionError("Access denied")
        
        output_dir = self._get_output_dir(task_id)
        files = []
        
        if output_dir.exists():
            for file_path in output_dir.rglob("*"):
                if file_path.is_file():
                    stat = file_path.stat()
                    files.append({
                        "path": str(file_path.relative_to(output_dir)),
                        "size": stat.st_size,
                        "modified_at": datetime.fromtimestamp(stat.st_mtime).isoformat(),
                    })
        
        return {
            "task_id": task_id,
            "files": files,
        }
    
    async def get_file(self, task_id: str, file_path: str) -> tuple[Path, str]:
        """Get a specific file from task outputs.
        
        Returns:
            Tuple of (file_path, filename).
        """
        task = await TaskDB.get(task_id)
        if not task:
            raise ValueError(f"Task not found: {task_id}")
        if task["user_id"] != self.user_id:
            raise PermissionError("Access denied")
        
        output_dir = self._get_output_dir(task_id)
        full_path = output_dir / file_path
        
        # Security check - ensure path is within output directory
        if not full_path.resolve().is_relative_to(output_dir.resolve()):
            raise ValueError("Invalid file path")
        
        if not full_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
        
        return full_path, full_path.name
    
    def _enrich_task(self, task: dict) -> dict:
        """Add computed fields to task."""
        # Task data is returned as-is, no additional enrichment needed
        return task
