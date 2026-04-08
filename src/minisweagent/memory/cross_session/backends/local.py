"""SQLite backend for cross-session memory.

Single file at ~/.cache/geak/memory.db (configurable).
WAL mode for concurrent reads.  No external dependencies.
"""

from __future__ import annotations

import json
import sqlite3
import threading
from pathlib import Path
from typing import Any

from minisweagent.memory.cross_session.schemas import ExperienceRecord, StrategySkill

_CREATE_EXPERIENCES = """
CREATE TABLE IF NOT EXISTS experiences (
    record_id        TEXT PRIMARY KEY,
    timestamp        TEXT NOT NULL,
    kernel_path      TEXT NOT NULL DEFAULT '',
    kernel_name      TEXT NOT NULL DEFAULT '',
    kernel_category  TEXT NOT NULL DEFAULT 'unknown',
    kernel_language  TEXT NOT NULL DEFAULT 'unknown',
    repo_url         TEXT NOT NULL DEFAULT '',
    bottleneck_type  TEXT NOT NULL DEFAULT 'unknown',
    baseline_latency_ms REAL NOT NULL DEFAULT 0,
    hardware         TEXT NOT NULL DEFAULT '',
    best_speedup     REAL NOT NULL DEFAULT 1.0,
    best_latency_ms  REAL NOT NULL DEFAULT 0,
    success          INTEGER NOT NULL DEFAULT 0,
    best_strategy    TEXT NOT NULL DEFAULT '',
    best_change_category TEXT NOT NULL DEFAULT '',
    key_insight      TEXT NOT NULL DEFAULT '',
    trajectory_sketch TEXT NOT NULL DEFAULT '',
    patch_file       TEXT NOT NULL DEFAULT '',
    final_report_path TEXT NOT NULL DEFAULT '',
    notebook_dir     TEXT NOT NULL DEFAULT '',
    -- JSON-encoded complex fields
    top_kernels      TEXT NOT NULL DEFAULT '[]',
    profiling_metrics TEXT NOT NULL DEFAULT '{}',
    what_worked      TEXT NOT NULL DEFAULT '[]',
    what_failed      TEXT NOT NULL DEFAULT '[]',
    dead_ends        TEXT NOT NULL DEFAULT '[]'
);
"""

_CREATE_SKILLS = """
CREATE TABLE IF NOT EXISTS skills (
    skill_id          TEXT PRIMARY KEY,
    title             TEXT NOT NULL DEFAULT '',
    change_category   TEXT NOT NULL DEFAULT '',
    strategy_description TEXT NOT NULL DEFAULT '',
    expected_speedup  TEXT NOT NULL DEFAULT '',
    evidence_count    INTEGER NOT NULL DEFAULT 0,
    success_rate      REAL NOT NULL DEFAULT 0,
    last_updated      TEXT NOT NULL DEFAULT '',
    -- JSON-encoded list fields
    kernel_categories TEXT NOT NULL DEFAULT '[]',
    bottleneck_types  TEXT NOT NULL DEFAULT '[]',
    kernel_languages  TEXT NOT NULL DEFAULT '[]',
    contraindications TEXT NOT NULL DEFAULT '[]',
    source_records    TEXT NOT NULL DEFAULT '[]'
);
"""

_CREATE_INDEXES = [
    "CREATE INDEX IF NOT EXISTS idx_exp_category ON experiences(kernel_category);",
    "CREATE INDEX IF NOT EXISTS idx_exp_bottleneck ON experiences(bottleneck_type);",
    "CREATE INDEX IF NOT EXISTS idx_exp_language ON experiences(kernel_language);",
    "CREATE INDEX IF NOT EXISTS idx_exp_hardware ON experiences(hardware);",
    "CREATE INDEX IF NOT EXISTS idx_exp_success ON experiences(success);",
    "CREATE INDEX IF NOT EXISTS idx_exp_timestamp ON experiences(timestamp);",
]

_JSON_FIELDS_EXP = {"top_kernels", "profiling_metrics", "what_worked", "what_failed", "dead_ends"}
_JSON_FIELDS_SKILL = {"kernel_categories", "bottleneck_types", "kernel_languages", "contraindications", "source_records"}


class LocalSQLiteBackend:
    """Thread-safe SQLite backend with WAL mode."""

    def __init__(self, db_path: str | None = None):
        if db_path is None:
            db_path = str(Path.home() / ".cache" / "geak" / "memory.db")
        self._db_path = db_path
        Path(self._db_path).parent.mkdir(parents=True, exist_ok=True)
        self._local = threading.local()
        self._init_db()

    def _get_conn(self) -> sqlite3.Connection:
        conn = getattr(self._local, "conn", None)
        if conn is None:
            conn = sqlite3.connect(self._db_path, timeout=10)
            conn.execute("PRAGMA journal_mode=WAL;")
            conn.execute("PRAGMA busy_timeout=5000;")
            conn.row_factory = sqlite3.Row
            self._local.conn = conn
        return conn

    def _init_db(self) -> None:
        conn = self._get_conn()
        conn.execute(_CREATE_EXPERIENCES)
        conn.execute(_CREATE_SKILLS)
        for idx_sql in _CREATE_INDEXES:
            conn.execute(idx_sql)
        conn.commit()

    # ── Experience CRUD ──────────────────────────────────────────────

    def store_experience(self, record: ExperienceRecord) -> None:
        d = record.to_dict()
        for k in _JSON_FIELDS_EXP:
            d[k] = json.dumps(d[k], ensure_ascii=False, default=str)
        d["success"] = int(d["success"])

        cols = ", ".join(d.keys())
        placeholders = ", ".join(f":{k}" for k in d)
        sql = f"INSERT OR REPLACE INTO experiences ({cols}) VALUES ({placeholders})"
        conn = self._get_conn()
        conn.execute(sql, d)
        conn.commit()

    def search_experiences(
        self,
        *,
        category: str | None = None,
        bottleneck: str | None = None,
        language: str | None = None,
        hardware: str | None = None,
        limit: int = 50,
    ) -> list[ExperienceRecord]:
        clauses: list[str] = []
        params: dict[str, Any] = {}
        if category:
            clauses.append("kernel_category = :cat")
            params["cat"] = category
        if bottleneck:
            clauses.append("bottleneck_type = :bn")
            params["bn"] = bottleneck
        if language:
            clauses.append("kernel_language = :lang")
            params["lang"] = language
        if hardware:
            clauses.append("hardware = :hw")
            params["hw"] = hardware

        where = (" WHERE " + " AND ".join(clauses)) if clauses else ""
        sql = f"SELECT * FROM experiences{where} ORDER BY timestamp DESC LIMIT :lim"
        params["lim"] = limit

        conn = self._get_conn()
        rows = conn.execute(sql, params).fetchall()
        return [self._row_to_experience(r) for r in rows]

    def list_experiences(self, *, limit: int = 100, offset: int = 0) -> list[ExperienceRecord]:
        conn = self._get_conn()
        rows = conn.execute(
            "SELECT * FROM experiences ORDER BY timestamp DESC LIMIT :lim OFFSET :off",
            {"lim": limit, "off": offset},
        ).fetchall()
        return [self._row_to_experience(r) for r in rows]

    def experience_count(self) -> int:
        conn = self._get_conn()
        return conn.execute("SELECT COUNT(*) FROM experiences").fetchone()[0]

    # ── Skill CRUD ───────────────────────────────────────────────────

    def store_skill(self, skill: StrategySkill) -> None:
        d = skill.to_dict()
        for k in _JSON_FIELDS_SKILL:
            d[k] = json.dumps(d[k], ensure_ascii=False, default=str)

        cols = ", ".join(d.keys())
        placeholders = ", ".join(f":{k}" for k in d)
        sql = f"INSERT OR REPLACE INTO skills ({cols}) VALUES ({placeholders})"
        conn = self._get_conn()
        conn.execute(sql, d)
        conn.commit()

    def search_skills(
        self,
        *,
        category: str | None = None,
        bottleneck: str | None = None,
        language: str | None = None,
        limit: int = 10,
    ) -> list[StrategySkill]:
        clauses: list[str] = []
        params: dict[str, Any] = {}
        if category:
            clauses.append("kernel_categories LIKE :cat")
            params["cat"] = f"%{category}%"
        if bottleneck:
            clauses.append("bottleneck_types LIKE :bn")
            params["bn"] = f"%{bottleneck}%"
        if language:
            clauses.append("kernel_languages LIKE :lang")
            params["lang"] = f"%{language}%"

        where = (" WHERE " + " AND ".join(clauses)) if clauses else ""
        sql = f"SELECT * FROM skills{where} ORDER BY evidence_count DESC, success_rate DESC LIMIT :lim"
        params["lim"] = limit

        conn = self._get_conn()
        rows = conn.execute(sql, params).fetchall()
        return [self._row_to_skill(r) for r in rows]

    def list_skills(self) -> list[StrategySkill]:
        conn = self._get_conn()
        rows = conn.execute("SELECT * FROM skills ORDER BY evidence_count DESC").fetchall()
        return [self._row_to_skill(r) for r in rows]

    # ── Stats ────────────────────────────────────────────────────────

    def get_stats(self) -> dict[str, Any]:
        conn = self._get_conn()
        exp_count = conn.execute("SELECT COUNT(*) FROM experiences").fetchone()[0]
        skill_count = conn.execute("SELECT COUNT(*) FROM skills").fetchone()[0]
        success_count = conn.execute("SELECT COUNT(*) FROM experiences WHERE success = 1").fetchone()[0]

        cat_rows = conn.execute(
            "SELECT kernel_category, COUNT(*) as cnt FROM experiences GROUP BY kernel_category ORDER BY cnt DESC"
        ).fetchall()
        bn_rows = conn.execute(
            "SELECT bottleneck_type, COUNT(*) as cnt FROM experiences GROUP BY bottleneck_type ORDER BY cnt DESC"
        ).fetchall()

        return {
            "experience_count": exp_count,
            "skill_count": skill_count,
            "success_count": success_count,
            "success_rate": round(success_count / exp_count, 3) if exp_count else 0,
            "categories": {r["kernel_category"]: r["cnt"] for r in cat_rows},
            "bottlenecks": {r["bottleneck_type"]: r["cnt"] for r in bn_rows},
        }

    # ── Helpers ──────────────────────────────────────────────────────

    @staticmethod
    def _row_to_experience(row: sqlite3.Row) -> ExperienceRecord:
        d = dict(row)
        for k in _JSON_FIELDS_EXP:
            if isinstance(d.get(k), str):
                try:
                    d[k] = json.loads(d[k])
                except (json.JSONDecodeError, TypeError):
                    pass
        d["success"] = bool(d.get("success", 0))
        return ExperienceRecord.from_dict(d)

    @staticmethod
    def _row_to_skill(row: sqlite3.Row) -> StrategySkill:
        d = dict(row)
        for k in _JSON_FIELDS_SKILL:
            if isinstance(d.get(k), str):
                try:
                    d[k] = json.loads(d[k])
                except (json.JSONDecodeError, TypeError):
                    pass
        return StrategySkill.from_dict(d)
