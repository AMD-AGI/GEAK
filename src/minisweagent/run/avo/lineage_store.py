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
    # Per-shape verified speedups for this version (#2: per-config signal that the
    # agent can compare across prior implementations). Empty when single-shape.
    per_shape: dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "parent_id": self.parent_id,
            "patch": self.patch,
            "git_ref": self.git_ref,
            "strategy": self.strategy,
            "score": {
                "speedup": self.speedup,
                "latency_ms": self.latency_ms,
                "verified": True,
                "per_shape": self.per_shape,
            },
            "committed_at": self.committed_at,
        }

    @classmethod
    def from_dict(cls, d: dict) -> LineageNode:
        score = d.get("score", {}) or {}
        per_shape = score.get("per_shape") or {}
        return cls(
            id=d["id"],
            parent_id=d.get("parent_id"),
            patch=d.get("patch"),
            git_ref=d.get("git_ref"),
            strategy=d.get("strategy"),
            speedup=float(score.get("speedup", 1.0)),
            latency_ms=score.get("latency_ms"),
            committed_at=d.get("committed_at", ""),
            per_shape={str(k): float(v) for k, v in per_shape.items()} if isinstance(per_shape, dict) else {},
        )


@dataclass
class LineageStore:
    """File-backed single-lineage store with a git-tagged commit gate."""

    state_dir: Path
    epsilon: float = 0.001
    language: str = "python"
    # Anti-"lazy optimization" floor (Kernel-Smith §3.3): a committed version
    # must EXCEED this verified speedup over the original baseline. Default 1.0
    # means a candidate must be genuinely faster than the baseline — a trivial
    # correct-but-no-gain (~1.0x) rewrite is rejected instead of entering the
    # lineage as the first "improvement".
    min_commit_speedup: float = 1.0
    # Per-shape regression guard (B2): reject a commit if ANY shape's verified
    # speedup falls below this floor, even when the geomean passes. Default 0.0
    # disables it (preserves single-number behavior); set e.g. 0.95 to forbid
    # commits that regress any shape by >5%.
    min_per_shape_speedup: float = 0.0
    # Measurement-significance margin (B1): to count as a genuine new best, a
    # candidate must exceed ``best * (1 + significance_margin)`` — a noise floor
    # *above* the current best, so a within-noise "tie" is not committed. Default
    # 0.0 reverts to the epsilon-tolerant "matches-or-improves" behavior.
    significance_margin: float = 0.0
    committed: list[LineageNode] = field(default_factory=list)
    # Explicit "tip" pointer. When set, ``best_node`` returns this node instead
    # of the global max-speedup node. Used by supervisor backtracking (P2) so a
    # run can resume exploration from an earlier committed version.
    active_best_id: str | None = None

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
            self.active_best_id = data.get("active_best_id")
            logger.info("LineageStore: resumed %d committed versions from %s", len(self.committed), self.lineage_path)
        except (json.JSONDecodeError, OSError, KeyError) as exc:
            logger.warning("LineageStore: failed to load %s (%s); starting empty.", self.lineage_path, exc)
            self.committed = []

    def _save(self) -> None:
        payload = {
            "best_id": self.best_id,
            "active_best_id": self.active_best_id,
            "committed": [n.to_dict() for n in self.committed],
        }
        self.lineage_path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")

    # ------------------------------------------------------------------
    # Lineage queries
    # ------------------------------------------------------------------

    @property
    def best_node(self) -> LineageNode | None:
        if not self.committed:
            return None
        if self.active_best_id:
            for node in self.committed:
                if node.id == self.active_best_id:
                    return node
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

    def top_k(self, k: int, *, exclude_baseline: bool = True, exclude_id: str | None = None) -> list[LineageNode]:
        """Return up to ``k`` committed versions by descending speedup.

        Used to inject multiple prior implementations into a step prompt (#2).
        Skips the baseline (v0, no patch) and an optional id (e.g. the current
        best already shown as the exemplar).
        """
        pool = [
            n
            for n in self.committed
            if (not exclude_baseline or n.patch)
            and n.id != exclude_id
        ]
        return sorted(pool, key=lambda n: n.speedup, reverse=True)[: max(0, k)]

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

    def seed_from_baseline(self, output_dir: Path, repo: Path | None = None) -> None:
        """Create ``v0`` from the preprocess baseline if the lineage is empty.

        When ``repo`` is a git repo, the current (pristine, post-preprocess) repo
        state is committed and tagged ``avo-v0`` so the lineage chain has a real
        base to apply incremental patches onto (A1).
        """
        if self.committed:
            return
        latency = self._read_baseline_latency(output_dir)
        if repo is not None:
            self._tag_baseline(repo)
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

    def _tag_baseline(self, repo: Path) -> None:
        """Commit + tag the current repo state as ``avo-v0`` (the lineage base)."""
        if not (Path(repo) / ".git").exists():
            logger.warning("LineageStore: %s is not a git repo; worktree reset/tagging disabled.", repo)
            return
        try:
            subprocess.run(["git", "-C", str(repo), "add", "-A"], check=False, capture_output=True)
            subprocess.run(
                ["git", "-C", str(repo), "commit", "-m", "avo: baseline (v0)", "--allow-empty"],
                check=False,
                capture_output=True,
            )
            subprocess.run(["git", "-C", str(repo), "tag", "-f", "avo-v0"], check=False, capture_output=True)
            logger.info("LineageStore: tagged baseline as avo-v0")
        except (OSError, subprocess.SubprocessError) as exc:
            logger.warning("LineageStore: baseline tagging failed: %s", exc)

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

        # Anti-lazy-optimization floor: must be genuinely faster than baseline.
        if candidate_speedup <= self.min_commit_speedup:
            logger.info(
                "commit gate: step %d speedup %.4fx not above lazy-opt floor %.4fx; not committed.",
                result.step_index,
                candidate_speedup,
                self.min_commit_speedup,
            )
            return False

        # Threshold vs current best: the epsilon tolerance permits near-ties,
        # while the B1 significance margin (when set) requires the candidate to
        # clear best by a noise floor. Take the stricter of the two.
        threshold = self.best_speedup * max(1.0 - self.epsilon, 1.0 + self.significance_margin)
        if candidate_speedup < threshold:
            logger.info(
                "commit gate: step %d speedup %.4fx below threshold %.4fx (best=%.4fx, margin=%.3f); not committed.",
                result.step_index,
                candidate_speedup,
                threshold,
                self.best_speedup,
                self.significance_margin,
            )
            return False

        # Per-shape regression guard (B2): geomean can hide a regressed shape.
        if self.min_per_shape_speedup > 0.0 and result.per_shape_speedups:
            weak = {s: v for s, v in result.per_shape_speedups.items() if v < self.min_per_shape_speedup}
            if weak:
                logger.info(
                    "commit gate: step %d rejected — %d shape(s) below per-shape floor %.3fx: %s",
                    result.step_index,
                    len(weak),
                    self.min_per_shape_speedup,
                    weak,
                )
                return False

        version_id = self._next_version_id()
        git_ref = f"avo-{version_id}"
        parent_ref = self.best_git_ref  # captured BEFORE appending the new node
        stored_patch = self._store_patch(result, version_id)
        if repo is not None and stored_patch is not None:
            self._materialize_and_tag(repo, parent_ref, stored_patch, git_ref)

        node = LineageNode(
            id=version_id,
            parent_id=self.best_id,
            patch=str(stored_patch) if stored_patch else None,
            git_ref=git_ref,
            strategy=result.strategy or (self.current_direction().get("strategy") or None),
            speedup=candidate_speedup,
            latency_ms=None,
            committed_at=_now_iso(),
            per_shape=dict(result.per_shape_speedups or {}),
        )
        self.committed.append(node)
        self.active_best_id = version_id  # advance the tip to the new commit
        self._save()
        logger.info("commit gate: committed %s (%.4fx) from step %d.", version_id, candidate_speedup, result.step_index)
        return True

    def commit_from_round(self, round_eval: Any, repo: Path | None = None) -> bool:
        """Fold a GEAK ``RoundEvaluation`` (e.g. from an ESCALATE rescue) into the lineage.

        Reuses :meth:`maybe_commit` so the same commit gate applies. Prefers the
        independently-verified FULL_BENCHMARK geomean speedup over the
        agent-reported one.
        """
        if round_eval is None:
            return False
        fb = getattr(round_eval, "full_benchmark", None)
        verified = fb.verified_speedup if fb is not None and getattr(fb, "verified_speedup", None) is not None else None
        if verified is None:
            verified = getattr(round_eval, "benchmark_speedup", None)
        patch = getattr(round_eval, "best_patch", "") or ""
        if not patch or verified is None:
            return False
        synthetic = VariationResult(
            step_index=-1,
            step_dir=Path(patch).parent,
            strategy="escalate",
            best_patch_path=Path(patch),
            best_speedup=float(verified),
            best_correct=True,
        )
        return self.maybe_commit(synthetic, repo=repo)

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

    def _materialize_and_tag(self, repo: Path, parent_ref: str | None, patch: Path, git_ref: str) -> None:
        """Make ``git_ref`` authoritatively equal the *verified* patch (A1).

        The agent's final worktree may hold a different (or worse) attempt than
        the verified best. So instead of committing the dirty worktree, we
        reconstruct the verified state: reset to the parent best, apply the
        verified incremental patch (the same diff ``evaluate_round_best`` used),
        then commit + tag. This guarantees ``avo-v{N}`` == the patch that was
        benchmarked, so the next step's ``reset_worktree_to_best`` starts from
        the real best.
        """
        if not (Path(repo) / ".git").exists():
            return
        base = parent_ref or "avo-v0"
        try:
            self._checkout(repo, base)  # clean parent-best worktree
            applied = self._git_apply(repo, patch)
            if not applied:
                logger.warning(
                    "LineageStore: git apply of %s onto %s failed; committing current worktree as fallback "
                    "(tag may diverge from verified patch).",
                    patch.name,
                    base,
                )
            subprocess.run(["git", "-C", str(repo), "add", "-A"], check=False, capture_output=True)
            subprocess.run(
                ["git", "-C", str(repo), "commit", "-m", f"avo: {git_ref}", "--allow-empty"],
                check=False,
                capture_output=True,
            )
            subprocess.run(["git", "-C", str(repo), "tag", "-f", git_ref], check=False, capture_output=True)
        except (OSError, subprocess.SubprocessError) as exc:
            logger.warning("LineageStore: materialize/tag %s failed: %s", git_ref, exc)

    @staticmethod
    def _git_apply(repo: Path, patch: Path) -> bool:
        """Apply ``patch`` (git-style relative diff). Try --3way, then plain."""
        for extra in (["--3way"], []):
            res = subprocess.run(
                ["git", "-C", str(repo), "apply", *extra, str(patch)],
                check=False,
                capture_output=True,
            )
            if res.returncode == 0:
                return True
        return False

    def reset_worktree_to_best(self, repo: Path) -> None:
        """Reset the repo worktree to the current best git ref before a step.

        No-op when there is no best ref yet (first step starts from baseline).
        """
        ref = self.best_git_ref
        if ref is None or ref == "avo-v0":
            return
        self._checkout(repo, ref)

    def set_best_pointer(self, version_id: str) -> bool:
        """Backtrack: move the active-best tip to an earlier committed version.

        Returns False if ``version_id`` is not a committed node. Subsequent
        commits gate against (and branch from) this node — single-lineage
        semantics, no archive/tree.
        """
        if any(n.id == version_id for n in self.committed):
            self.active_best_id = version_id
            self._save()
            logger.info("LineageStore: best pointer backtracked to %s", version_id)
            return True
        logger.warning("LineageStore: backtrack target %s not found; ignored.", version_id)
        return False

    def reset_worktree_to(self, repo: Path, version_id: str) -> None:
        """Checkout the worktree to a specific committed version's git ref."""
        node = next((n for n in self.committed if n.id == version_id), None)
        if node is None or not node.git_ref:
            logger.warning("LineageStore: cannot reset worktree to %s (unknown or untagged).", version_id)
            return
        self._checkout(repo, node.git_ref)

    def _checkout(self, repo: Path, ref: str) -> None:
        try:
            subprocess.run(["git", "-C", str(repo), "checkout", "-f", ref], check=False, capture_output=True)
            # A2: drop untracked junk from a prior aborted step so it can't leak
            # into the next step's diff. Tracked files (incl. anything captured by
            # a prior commit's `git add -A`) are preserved by checkout -f.
            subprocess.run(["git", "-C", str(repo), "clean", "-fd"], check=False, capture_output=True)
            logger.info("LineageStore: worktree reset to %s (clean)", ref)
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
