"""Background task scheduler for GEAK Online Service.

Periodically checks running tasks and updates their status.
"""

import asyncio
import logging
import shutil
from datetime import datetime, timedelta
from pathlib import Path

from server.database import TaskDB
from server.services.safe_client import SaFEClient
from server.config import get_settings

logger = logging.getLogger(__name__)


class TaskScheduler:
    """Background scheduler for task status updates."""
    
    def __init__(self):
        self.settings = get_settings()
        self._running = False
        self._task: asyncio.Task | None = None
        self.check_interval = 60  # Check every 60 seconds
        self.cleanup_interval = 3600  # Cleanup every 1 hour
        self.task_retention_days = 7  # Keep tasks for 7 days
        self._loops_since_cleanup = 0
    
    async def start(self):
        """Start the background scheduler."""
        if self._running:
            return
        
        self._running = True
        self._task = asyncio.create_task(self._run_loop())
        logger.info("Task scheduler started (interval: %ds)", self.check_interval)
    
    async def stop(self):
        """Stop the background scheduler."""
        self._running = False
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
        logger.info("Task scheduler stopped")
    
    async def _run_loop(self):
        """Main scheduler loop."""
        cleanup_every_n = max(1, self.cleanup_interval // self.check_interval)
        
        while self._running:
            try:
                await self._check_running_tasks()
            except Exception as e:
                logger.exception("Error in task scheduler: %s", e)
            
            self._loops_since_cleanup += 1
            if self._loops_since_cleanup >= cleanup_every_n:
                self._loops_since_cleanup = 0
                try:
                    await self._cleanup_expired_tasks()
                except Exception as e:
                    logger.exception("Error in task cleanup: %s", e)
            
            await asyncio.sleep(self.check_interval)
    
    async def _check_running_tasks(self):
        """Check all running tasks and update their status."""
        # Get all running tasks from database
        running_tasks = await self._get_all_running_tasks()
        
        if not running_tasks:
            return
        
        logger.info("Checking %d running tasks", len(running_tasks))
        
        for task in running_tasks:
            try:
                await self._update_task_status(task)
            except Exception as e:
                logger.warning("Error updating task %s: %s", task["id"], e)
    
    async def _get_all_running_tasks(self) -> list[dict]:
        """Get all tasks with running status from database."""
        from server.database import get_db_path
        import aiosqlite
        
        db_path = get_db_path()
        async with aiosqlite.connect(db_path) as db:
            db.row_factory = aiosqlite.Row
            cursor = await db.execute(
                "SELECT * FROM tasks WHERE status = 'running'"
            )
            rows = await cursor.fetchall()
            return [dict(row) for row in rows]
    
    async def _update_task_status(self, task: dict):
        """Update a single task's status by querying SaFE workload."""
        task_id = task["id"]
        workload_id = task.get("safe_workload_id")
        
        if not workload_id:
            logger.warning("Task %s has no workload ID", task_id)
            return
        
        # We need an API key to query SaFE. Use a system-level key or skip if not available
        # For now, we'll try to get workload status without user context
        try:
            # Create a client with the system API key (if configured)
            system_api_key = self.settings.safe_system_api_key
            if not system_api_key:
                # Fallback: check log file for completion
                await self._check_log_for_completion(task)
                return
            
            client = SaFEClient(system_api_key)
            workload = await client.get_workload(workload_id)
            
            # Map SaFE workload status to task status
            safe_status = workload.get("status", "").lower()
            phase = workload.get("phase", "").lower()
            
            new_status = self._map_workload_status(safe_status, phase)
            
            if new_status and new_status != task["status"]:
                await TaskDB.update(task_id, status=new_status)
                logger.info("Task %s status updated: %s -> %s", task_id, task["status"], new_status)
        
        except Exception as e:
            logger.debug("Could not query workload %s: %s, falling back to log check", workload_id, e)
            # Fallback to log-based detection
            await self._check_log_for_completion(task)
    
    def _map_workload_status(self, status: str, phase: str) -> str | None:
        """Map SaFE workload status to task status.
        
        SaFE workload statuses:
        - Pending, Running, Succeeded, Failed, Unknown
        """
        # Check phase first (more specific)
        if phase in ("succeeded", "completed"):
            return "completed"
        elif phase == "failed":
            return "failed"
        elif phase in ("pending", "creating"):
            return "running"  # Keep as running
        
        # Check status
        if status == "succeeded":
            return "completed"
        elif status == "failed":
            return "failed"
        elif status in ("pending", "running"):
            return "running"
        
        return None  # Unknown, don't update
    
    async def _check_log_for_completion(self, task: dict):
        """Fallback: Check execution log for completion markers."""
        task_id = task["id"]
        user_id = task.get("user_id", "")
        
        output_dir = Path(self.settings.nfs_base_path) / "tasks" / user_id / task_id / "output"
        log_path = output_dir / "execution.log"
        
        if not log_path.exists():
            return
        
        try:
            content = log_path.read_text()
            
            if "completed successfully" in content:
                await TaskDB.update(task_id, status="completed")
                logger.info("Task %s marked completed (log check)", task_id)
            elif "error" in content.lower() and ("fatal" in content.lower() or "exception" in content.lower()):
                pass  # Be conservative, don't auto-fail
        except Exception as e:
            logger.debug("Could not read log for task %s: %s", task_id, e)
    
    async def _cleanup_expired_tasks(self):
        """Delete tasks and their directories older than retention period."""
        cutoff = datetime.utcnow() - timedelta(days=self.task_retention_days)
        expired_tasks = await TaskDB.list_expired(cutoff)
        
        if not expired_tasks:
            logger.debug("No expired tasks to clean up")
            return
        
        logger.info("Cleaning up %d expired tasks (before %s)", len(expired_tasks), cutoff.isoformat())
        deleted_count = 0
        
        for task in expired_tasks:
            task_id = task["id"]
            user_id = task.get("user_id", "")
            task_dir = Path(self.settings.nfs_base_path) / "tasks" / user_id / task_id
            
            try:
                if task_dir.exists():
                    shutil.rmtree(task_dir)
                    deleted_count += 1
                    logger.info("Deleted task directory: %s (status=%s)", task_dir, task.get("status"))
                else:
                    logger.debug("Task directory not found, skipping: %s", task_dir)
            except Exception as e:
                logger.error("Failed to clean up task %s: %s", task_id, e)
        
        logger.info("Cleanup finished: %d/%d tasks deleted", deleted_count, len(expired_tasks))


# Global scheduler instance
_scheduler: TaskScheduler | None = None


def get_scheduler() -> TaskScheduler:
    """Get the global scheduler instance."""
    global _scheduler
    if _scheduler is None:
        _scheduler = TaskScheduler()
    return _scheduler


async def start_scheduler():
    """Start the global scheduler."""
    scheduler = get_scheduler()
    await scheduler.start()


async def stop_scheduler():
    """Stop the global scheduler."""
    scheduler = get_scheduler()
    await scheduler.stop()
