"""LineageStore — persistence for the AVO committed lineage ``P_t``.

The store owns everything under ``<output_dir>/avo_state/``:

- ``lineage.json``        — the committed version chain (``P_t``) + ``best_id``.
- ``attempts.jsonl``      — every attempt, including failures (the internal
                            search trajectory that is *not* committed).
- ``direction.json``      — the strategy currently assigned to the next step.
- ``heartbeat.json``      — liveness + resume anchor.

The **commit gate** (paper §3.2) lives here: a candidate enters the lineage iff
correctness passed *and* its independently-verified speedup is within
``epsilon`` of the running best. Non-improving / failing attempts are appended
to ``attempts.jsonl`` only.

Git is used the same way GEAK already relies on it: each committed version is
tagged ``avo-v{N}`` so a variation step can ``reset`` the worktree to the
current best before exploring.
"""

from __future__ import annotations

import json
import logging
import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from minisweagent.run.avo.result import VariationResult

logger = logging.getLogger(__name__)


@dataclass
class LineageNode:
    """One committed version in ``P_t``."""

    id: str
    parent_id: str | None
    patch: str | None
    git_ref: str | None
    strategy: str | None
    speedup: float
    latency_ms: float | None
    committed_at: str

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "parent_id": self.parent_id,
            "patch": self.patch,
            "git_ref": self.git_ref,
            "strategy": self.strategy,
            "score": {"speedup": self.speedup, "latency_ms": self.latency_ms, "verified": True},
            "committed_at": self.committed_at,
        }

    @classmethod
    def from_dict(cls, d: dict) -> LineageNode:
        score = d.get("score", {}) or {}
        return cls(
            id=d["id"],
            parent_id=d.get("parent_id"),
            patch=d.get("patch"),
            git_ref=d.get("git_ref"),
            strategy=d.get("strategy"),
            speedup=float(score.get("speedup", 1.0)),
            latency_ms=score.get("latency_ms"),
            committed_at=d.get("committed_at", ""),
        )


@dataclass
class LineageStore:
    """File-backed single-lineage store with a git-tagged commit gate."""

    state_dir: Path
    epsilon: float = 0.001
    language: str = "python"
    committed: list[LineageNode] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.state_dir = Path(self.state_dir)
        self.state_dir.mkdir(parents=True, exist_ok=True)
        self.patches_dir = self.state_dir / "patches"
        self.patches_dir.mkdir(parents=True, exist_ok=True)
        self.lineage_path = self.state_dir / "lineage.json"
        self.attempts_path = self.state_dir / "attempts.jsonl"
        self.direction_path = self.state_dir / "direction.json"
        self.heartbeat_path = self.state_dir / "heartbeat.json"
        if self.lineage_path.exists():
            self._load()

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def _load(self) -> None:
        try:
            data = json.loads(self.lineage_path.read_text(encoding="utf-8"))
            self.committed = [LineageNode.from_dict(d) for d in data.get("committed", [])]
            logger.info("LineageStore: resumed %d committed versions from %s", len(self.committed), self.lineage_path)
        except (json.JSONDecodeError, OSError, KeyError) as exc:
            logger.warning("LineageStore: failed to load %s (%s); starting empty.", self.lineage_path, exc)
            self.committed = []

    def _save(self) -> None:
        payload = {"best_id": self.best_id, "committed": [n.to_dict() for n in self.committed]}
        self.lineage_path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")

    # ------------------------------------------------------------------
    # Lineage queries
    # ------------------------------------------------------------------

    @property
    def best_node(self) -> LineageNode | None:
        if not self.committed:
            return None
        return max(self.committed, key=lambda n: n.speedup)

    @property
    def best_id(self) -> str | None:
        node = self.best_node
        return node.id if node else None

    @property
    def best_speedup(self) -> float:
        node = self.best_node
        return node.speedup if node else 1.0

    @property
    def best_git_ref(self) -> str | None:
        node = self.best_node
        return node.git_ref if node else None

    def _next_version_id(self) -> str:
        return f"v{len(self.committed)}"

    def summary(self, last_n: int = 5) -> str:
        """Compact lineage summary for prompt injection / supervisor bundles."""
        if not self.committed:
            return "(empty lineage; no committed versions yet)"
        tail = self.committed[-last_n:]
        chain = " -> ".join(f"{n.id} {n.strategy or '?'} {n.speedup:.3f}x" for n in tail)
        return f"best={self.best_id} ({self.best_speedup:.3f}x); recent: {chain}"

    # ------------------------------------------------------------------
    # Seeding (v0 baseline)
    # ------------------------------------------------------------------

    def seed_from_baseline(self, output_dir: Path) -> None:
        """Create ``v0`` from the preprocess baseline if the lineage is empty."""
        if self.committed:
            return
        latency = self._read_baseline_latency(output_dir)
        node = LineageNode(
            id="v0",
            parent_id=None,
            patch=None,
            git_ref="avo-v0",
            strategy="baseline",
            speedup=1.0,
            latency_ms=latency,
            committed_at=_now_iso(),
        )
        self.committed.append(node)
        self._save()
        logger.info("LineageStore: seeded v0 baseline (latency_ms=%s)", latency)

    @staticmethod
    def _read_baseline_latency(output_dir: Path) -> float | None:
        bm = Path(output_dir) / "baseline_metrics.json"
        if not bm.exists():
            return None
        try:
            data = json.loads(bm.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            return None
        for key in ("latency_ms", "baseline_ms", "wall_ms", "kernel_ms"):
            val = data.get(key)
            if isinstance(val, (int, float)) and val > 0:
                return float(val)
        return None

    # ------------------------------------------------------------------
    # Direction (current assigned strategy)
    # ------------------------------------------------------------------

    def current_direction(self) -> dict[str, Any]:
        if self.direction_path.exists():
            try:
                return json.loads(self.direction_path.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, OSError):
                pass
        return {"strategy": "", "assigned_by": "default", "supervisor_cycle": 0}

    def set_direction(self, strategy: str, *, assigned_by: str, supervisor_cycle: int) -> None:
        self.direction_path.write_text(
            json.dumps(
                {"strategy": strategy, "assigned_by": assigned_by, "supervisor_cycle": supervisor_cycle},
                indent=2,
            ),
            encoding="utf-8",
        )

    # ------------------------------------------------------------------
    # Recording + the commit gate
    # ------------------------------------------------------------------

    def record_attempts(self, result: VariationResult) -> None:
        """Append every attempt (incl. failures) to ``attempts.jsonl``."""
        with open(self.attempts_path, "a", encoding="utf-8") as fh:
            for attempt in result.attempts:
                row = attempt.to_dict()
                row["step_index"] = result.step_index
                fh.write(json.dumps(row, default=str) + "\n")

    def maybe_commit(self, result: VariationResult, repo: Path | None = None) -> bool:
        """Apply the commit gate. Returns True iff a new version was committed.

        Gate (paper §3.2):
          1. correctness passed, AND
          2. ``best_speedup >= running_best * (1 - epsilon)``.
        """
        if not result.produced_verified_improvement_candidate:
            logger.info("commit gate: step %d produced no verified correct candidate.", result.step_index)
            return False

        candidate_speedup = float(result.best_speedup)  # type: ignore[arg-type]
        threshold = self.best_speedup * (1.0 - self.epsilon)
        if candidate_speedup < threshold:
            logger.info(
                "commit gate: step %d speedup %.4fx below threshold %.4fx (best=%.4fx); not committed.",
                result.step_index,
                candidate_speedup,
                threshold,
                self.best_speedup,
            )
            return False

        version_id = self._next_version_id()
        git_ref = f"avo-{version_id}"
        stored_patch = self._store_patch(result, version_id)
        if repo is not None and stored_patch is not None:
            self._tag_git(repo, git_ref)

        node = LineageNode(
            id=version_id,
            parent_id=self.best_id,
            patch=str(stored_patch) if stored_patch else None,
            git_ref=git_ref,
            strategy=result.strategy or (self.current_direction().get("strategy") or None),
            speedup=candidate_speedup,
            latency_ms=None,
            committed_at=_now_iso(),
        )
        self.committed.append(node)
        self._save()
        logger.info("commit gate: committed %s (%.4fx) from step %d.", version_id, candidate_speedup, result.step_index)
        return True

    def _store_patch(self, result: VariationResult, version_id: str) -> Path | None:
        if result.best_patch_path is None or not Path(result.best_patch_path).exists():
            return None
        dest = self.patches_dir / f"{version_id}.patch"
        try:
            dest.write_text(Path(result.best_patch_path).read_text(encoding="utf-8"), encoding="utf-8")
            return dest
        except OSError as exc:
            logger.warning("LineageStore: failed to copy patch for %s: %s", version_id, exc)
            return None

    # ------------------------------------------------------------------
    # Git helpers (worktree reset + tagging)
    # ------------------------------------------------------------------

    def _tag_git(self, repo: Path, git_ref: str) -> None:
        """Commit the current worktree state and tag it ``git_ref``.

        Best-effort: a failure here does not invalidate the lineage entry
        (the ``.patch`` file remains the source of truth).
        """
        try:
            subprocess.run(["git", "-C", str(repo), "add", "-A"], check=False, capture_output=True)
            subprocess.run(
                ["git", "-C", str(repo), "commit", "-m", f"avo: {git_ref}", "--allow-empty"],
                check=False,
                capture_output=True,
            )
            subprocess.run(["git", "-C", str(repo), "tag", "-f", git_ref], check=False, capture_output=True)
        except (OSError, subprocess.SubprocessError) as exc:
            logger.warning("LineageStore: git tag %s failed: %s", git_ref, exc)

    def reset_worktree_to_best(self, repo: Path) -> None:
        """Reset the repo worktree to the current best git ref before a step.

        No-op when there is no best ref yet (first step starts from baseline).
        """
        ref = self.best_git_ref
        if ref is None or ref == "avo-v0":
            return
        try:
            subprocess.run(["git", "-C", str(repo), "checkout", "-f", ref], check=False, capture_output=True)
            logger.info("LineageStore: worktree reset to %s", ref)
        except (OSError, subprocess.SubprocessError) as exc:
            logger.warning("LineageStore: worktree reset to %s failed: %s", ref, exc)

    # ------------------------------------------------------------------
    # Heartbeat + finalize handoff
    # ------------------------------------------------------------------

    def heartbeat(self, *, step_index: int, extra: dict[str, Any] | None = None) -> None:
        payload = {
            "ts": time.time(),
            "step_index": step_index,
            "best_id": self.best_id,
            "best_speedup": self.best_speedup,
            "committed": len(self.committed),
        }
        if extra:
            payload.update(extra)
        self.heartbeat_path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")

    def build_postprocess_ctx(self, output_dir: Path) -> dict[str, Any]:
        """Build the ctx dict ``auto_finalize`` expects.

        AVO writes its own best patch into ``results/round_1/avo-best/`` so the
        existing ``auto_finalize`` scanner can pick it up without modification.
        """
        return {
            "output_dir": str(output_dir),
            "preprocess_dir": str(output_dir),
            "starting_patch": (self.best_node.patch if self.best_node else "") or "",
            "_best_global_speedup": self.best_speedup,
            "best_speedup": self.best_speedup,
            "best_id": self.best_id,
        }


def _now_iso() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
