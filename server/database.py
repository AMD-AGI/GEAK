"""Database module for GEAK Online Service."""

import aiosqlite
import json
from datetime import datetime
from pathlib import Path

from server.config import get_settings

# Database schema
SCHEMA = """
CREATE TABLE IF NOT EXISTS tasks (
    id TEXT PRIMARY KEY,
    user_id TEXT NOT NULL,
    status TEXT NOT NULL DEFAULT 'pending',
    input_type TEXT NOT NULL,
    input_path TEXT,
    prompt TEXT,
    config TEXT,
    runtime_config TEXT,
    safe_workload_id TEXT,
    output_path TEXT,
    error_message TEXT,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_tasks_user_id ON tasks(user_id);
CREATE INDEX IF NOT EXISTS idx_tasks_status ON tasks(status);
CREATE INDEX IF NOT EXISTS idx_tasks_created_at ON tasks(created_at);

CREATE TABLE IF NOT EXISTS user_configs (
    user_id TEXT PRIMARY KEY,
    model_config TEXT,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);
"""

# Global database path cache
_db_path: Path | None = None


def get_db_path() -> Path:
    """Get database file path, ensuring directory exists."""
    global _db_path
    if _db_path is None:
        settings = get_settings()
        _db_path = Path(settings.database_path)
        _db_path.parent.mkdir(parents=True, exist_ok=True)
    return _db_path


async def init_db():
    """Initialize database with schema."""
    db_path = get_db_path()
    async with aiosqlite.connect(db_path) as db:
        await db.executescript(SCHEMA)
        await db.commit()


class TaskDB:
    """Task database operations."""
    
    @staticmethod
    async def create(
        task_id: str,
        user_id: str,
        input_type: str,
        input_path: str | None = None,
        prompt: str | None = None,
        config: dict | None = None,
        runtime_config: dict | None = None,
    ) -> dict:
        """Create a new task."""
        now = datetime.utcnow().isoformat()
        db_path = get_db_path()
        async with aiosqlite.connect(db_path) as db:
            await db.execute(
                """
                INSERT INTO tasks (
                    id, user_id, status, input_type, input_path, 
                    prompt, config, runtime_config, created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    task_id,
                    user_id,
                    "pending",
                    input_type,
                    input_path,
                    prompt,
                    json.dumps(config) if config else None,
                    json.dumps(runtime_config) if runtime_config else None,
                    now,
                    now,
                ),
            )
            await db.commit()
        return await TaskDB.get(task_id)
    
    @staticmethod
    async def get(task_id: str) -> dict | None:
        """Get task by ID."""
        db_path = get_db_path()
        async with aiosqlite.connect(db_path) as db:
            db.row_factory = aiosqlite.Row
            cursor = await db.execute(
                "SELECT * FROM tasks WHERE id = ?",
                (task_id,),
            )
            row = await cursor.fetchone()
            if row:
                return TaskDB._row_to_dict(row)
            return None
    
    @staticmethod
    async def list_by_user(
        user_id: str,
        status: str | None = None,
        limit: int = 20,
        offset: int = 0,
    ) -> list[dict]:
        """List tasks for a user."""
        db_path = get_db_path()
        async with aiosqlite.connect(db_path) as db:
            db.row_factory = aiosqlite.Row
            if status:
                cursor = await db.execute(
                    """
                    SELECT * FROM tasks 
                    WHERE user_id = ? AND status = ?
                    ORDER BY created_at DESC
                    LIMIT ? OFFSET ?
                    """,
                    (user_id, status, limit, offset),
                )
            else:
                cursor = await db.execute(
                    """
                    SELECT * FROM tasks 
                    WHERE user_id = ?
                    ORDER BY created_at DESC
                    LIMIT ? OFFSET ?
                    """,
                    (user_id, limit, offset),
                )
            rows = await cursor.fetchall()
            return [TaskDB._row_to_dict(row) for row in rows]
    
    @staticmethod
    async def update(task_id: str, **kwargs) -> dict | None:
        """Update task fields."""
        if not kwargs:
            return await TaskDB.get(task_id)
        
        # Handle JSON fields
        for field in ["config", "runtime_config"]:
            if field in kwargs and isinstance(kwargs[field], dict):
                kwargs[field] = json.dumps(kwargs[field])
        
        kwargs["updated_at"] = datetime.utcnow().isoformat()
        
        set_clause = ", ".join(f"{k} = ?" for k in kwargs.keys())
        values = list(kwargs.values()) + [task_id]
        
        db_path = get_db_path()
        async with aiosqlite.connect(db_path) as db:
            await db.execute(
                f"UPDATE tasks SET {set_clause} WHERE id = ?",
                values,
            )
            await db.commit()
        
        return await TaskDB.get(task_id)
    
    @staticmethod
    async def list_expired(before: datetime) -> list[dict]:
        """List all tasks older than a given time, regardless of status."""
        db_path = get_db_path()
        async with aiosqlite.connect(db_path) as db:
            db.row_factory = aiosqlite.Row
            cursor = await db.execute(
                """
                SELECT * FROM tasks
                WHERE updated_at < ?
                ORDER BY updated_at ASC
                """,
                (before.isoformat(),),
            )
            rows = await cursor.fetchall()
            return [TaskDB._row_to_dict(row) for row in rows]

    @staticmethod
    async def delete(task_id: str) -> bool:
        """Delete a task."""
        db_path = get_db_path()
        async with aiosqlite.connect(db_path) as db:
            cursor = await db.execute(
                "DELETE FROM tasks WHERE id = ?",
                (task_id,),
            )
            await db.commit()
            return cursor.rowcount > 0
    
    @staticmethod
    def _row_to_dict(row: aiosqlite.Row) -> dict:
        """Convert database row to dictionary."""
        data = dict(row)
        # Parse JSON fields
        for field in ["config", "runtime_config"]:
            if data.get(field):
                try:
                    data[field] = json.loads(data[field])
                except json.JSONDecodeError:
                    pass
        return data


class UserConfigDB:
    """User configuration database operations."""
    
    @staticmethod
    async def get(user_id: str) -> dict | None:
        """Get user's model configuration."""
        db_path = get_db_path()
        async with aiosqlite.connect(db_path) as db:
            db.row_factory = aiosqlite.Row
            cursor = await db.execute(
                "SELECT * FROM user_configs WHERE user_id = ?",
                (user_id,),
            )
            row = await cursor.fetchone()
            if row:
                return UserConfigDB._row_to_dict(row)
            return None
    
    @staticmethod
    async def upsert(user_id: str, model_config: dict) -> dict:
        """Create or update user's model configuration."""
        now = datetime.utcnow().isoformat()
        db_path = get_db_path()
        async with aiosqlite.connect(db_path) as db:
            # Check if exists
            cursor = await db.execute(
                "SELECT user_id FROM user_configs WHERE user_id = ?",
                (user_id,),
            )
            exists = await cursor.fetchone()
            
            if exists:
                await db.execute(
                    """
                    UPDATE user_configs 
                    SET model_config = ?, updated_at = ?
                    WHERE user_id = ?
                    """,
                    (json.dumps(model_config), now, user_id),
                )
            else:
                await db.execute(
                    """
                    INSERT INTO user_configs (user_id, model_config, created_at, updated_at)
                    VALUES (?, ?, ?, ?)
                    """,
                    (user_id, json.dumps(model_config), now, now),
                )
            await db.commit()
        return await UserConfigDB.get(user_id)
    
    @staticmethod
    async def delete(user_id: str) -> bool:
        """Delete user's model configuration."""
        db_path = get_db_path()
        async with aiosqlite.connect(db_path) as db:
            cursor = await db.execute(
                "DELETE FROM user_configs WHERE user_id = ?",
                (user_id,),
            )
            await db.commit()
            return cursor.rowcount > 0
    
    @staticmethod
    def _row_to_dict(row: aiosqlite.Row) -> dict:
        """Convert database row to dictionary."""
        data = dict(row)
        if data.get("model_config"):
            try:
                data["model_config"] = json.loads(data["model_config"])
            except json.JSONDecodeError:
                pass
        return data
