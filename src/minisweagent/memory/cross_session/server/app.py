"""FastAPI server for shared cross-session memory.

Wraps a LocalSQLiteBackend behind a REST API so multiple developers
and CI pipelines can share optimization knowledge.

Run: uvicorn minisweagent.memory.cross_session.server.app:app --port 8642
Or:  geak memory serve --port 8642
"""

from __future__ import annotations

import os
from typing import Any

from fastapi import Depends, FastAPI, Header, HTTPException, Query
from pydantic import BaseModel

app = FastAPI(
    title="GEAK Cross-Session Memory",
    version="1.0.0",
    description="Shared optimization memory for kernel developers",
)

_EXPECTED_KEY = os.environ.get("GEAK_MEMORY_API_KEY", "").strip()
_DB_PATH = os.environ.get("GEAK_MEMORY_STORE_PATH", "/data/memory.db")


def _get_backend():
    from minisweagent.memory.cross_session.backends.local import LocalSQLiteBackend

    if not hasattr(app.state, "backend"):
        app.state.backend = LocalSQLiteBackend(db_path=_DB_PATH)
    return app.state.backend


def _check_auth(authorization: str | None = Header(default=None)):
    if not _EXPECTED_KEY:
        return
    if not authorization:
        raise HTTPException(status_code=401, detail="Missing Authorization header")
    token = authorization.removeprefix("Bearer ").strip()
    if token != _EXPECTED_KEY:
        raise HTTPException(status_code=403, detail="Invalid API key")


class ExperienceIn(BaseModel):
    class Config:
        extra = "allow"


class SkillIn(BaseModel):
    class Config:
        extra = "allow"


# ── Experience endpoints ─────────────────────────────────────────────

@app.post("/api/v1/experiences", dependencies=[Depends(_check_auth)])
def create_experience(body: ExperienceIn, backend=Depends(_get_backend)):
    from minisweagent.memory.cross_session.schemas import ExperienceRecord

    record = ExperienceRecord.from_dict(body.model_dump())
    backend.store_experience(record)
    return {"status": "ok", "record_id": record.record_id}


@app.get("/api/v1/experiences", dependencies=[Depends(_check_auth)])
def list_experiences(
    limit: int = Query(100, ge=1, le=1000),
    offset: int = Query(0, ge=0),
    backend=Depends(_get_backend),
):
    exps = backend.list_experiences(limit=limit, offset=offset)
    return {"results": [e.to_dict() for e in exps], "count": len(exps)}


@app.get("/api/v1/experiences/search", dependencies=[Depends(_check_auth)])
def search_experiences(
    category: str | None = None,
    bottleneck: str | None = None,
    language: str | None = None,
    hardware: str | None = None,
    limit: int = Query(50, ge=1, le=500),
    backend=Depends(_get_backend),
):
    exps = backend.search_experiences(
        category=category,
        bottleneck=bottleneck,
        language=language,
        hardware=hardware,
        limit=limit,
    )
    return {"results": [e.to_dict() for e in exps], "count": len(exps)}


# ── Skill endpoints ──────────────────────────────────────────────────

@app.post("/api/v1/skills", dependencies=[Depends(_check_auth)])
def create_skill(body: SkillIn, backend=Depends(_get_backend)):
    from minisweagent.memory.cross_session.schemas import StrategySkill

    skill = StrategySkill.from_dict(body.model_dump())
    backend.store_skill(skill)
    return {"status": "ok", "skill_id": skill.skill_id}


@app.get("/api/v1/skills", dependencies=[Depends(_check_auth)])
def list_skills(backend=Depends(_get_backend)):
    skills = backend.list_skills()
    return {"results": [s.to_dict() for s in skills], "count": len(skills)}


@app.get("/api/v1/skills/search", dependencies=[Depends(_check_auth)])
def search_skills(
    category: str | None = None,
    bottleneck: str | None = None,
    language: str | None = None,
    limit: int = Query(10, ge=1, le=100),
    backend=Depends(_get_backend),
):
    skills = backend.search_skills(
        category=category,
        bottleneck=bottleneck,
        language=language,
        limit=limit,
    )
    return {"results": [s.to_dict() for s in skills], "count": len(skills)}


# ── Stats & consolidation ────────────────────────────────────────────

@app.get("/api/v1/stats", dependencies=[Depends(_check_auth)])
def get_stats(backend=Depends(_get_backend)):
    return backend.get_stats()


@app.post("/api/v1/consolidate", dependencies=[Depends(_check_auth)])
def run_consolidation(backend=Depends(_get_backend)):
    from minisweagent.memory.cross_session.consolidation import consolidate

    skills = consolidate(backend)
    return {"status": "ok", "skills_produced": len(skills)}


@app.get("/api/v1/health")
def health():
    return {"status": "ok"}
