"""Cross-session memory for GEAK kernel optimization agent.

Stores optimization outcomes in a SQLite database so future runs can
learn from past successes and failures. Inspired by:
- SEDM (NeurIPS 2025): Verifiable write admission, cross-domain knowledge diffusion
- Cross-Task Experience Learning (ICLR 2026): Source experience memory

The database stores:
1. Optimization outcomes (strategy, speedup, success/failure per kernel category)
2. COMMANDMENT patterns (working templates per kernel category)
3. Known pitfalls (failure reasons to avoid)
"""

from __future__ import annotations

import json
import sqlite3
import hashlib
from datetime import datetime
from pathlib import Path
from typing import Any

from minisweagent import global_config_dir

DEFAULT_DB_PATH = global_config_dir / "geak_memory.db"


class CrossSessionMemory:
    """SQLite-backed cross-session memory for kernel optimization outcomes."""

    def __init__(self, db_path: str | Path | None = None):
        self.db_path = Path(db_path) if db_path else DEFAULT_DB_PATH
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(self.db_path))
        self._conn.row_factory = sqlite3.Row
        self._init_tables()
        self._migrate_columns()

    def _init_tables(self):
        self._conn.executescript("""
            CREATE TABLE IF NOT EXISTS optimization_outcomes (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                kernel_type TEXT,
                kernel_category TEXT,
                bottleneck_type TEXT,
                gpu_architecture TEXT,
                strategy_name TEXT,
                speedup_achieved REAL,
                success INTEGER,
                failure_reason TEXT,
                cost_dollars REAL,
                steps_taken INTEGER,
                commandment_worked INTEGER,
                profiling_metrics JSON,
                kernel_signature_hash TEXT,
                optimization_technique TEXT DEFAULT '',
                kernel_language TEXT DEFAULT '',
                patch_snippet TEXT DEFAULT '',
                timestamp TEXT DEFAULT CURRENT_TIMESTAMP
            );

            CREATE TABLE IF NOT EXISTS commandment_patterns (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                kernel_category TEXT,
                template_name TEXT,
                commandment_content TEXT,
                success_count INTEGER DEFAULT 0,
                failure_count INTEGER DEFAULT 0,
                last_used TEXT DEFAULT CURRENT_TIMESTAMP
            );

            CREATE TABLE IF NOT EXISTS known_pitfalls (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                kernel_category TEXT,
                pitfall_type TEXT,
                description TEXT,
                occurrences INTEGER DEFAULT 1,
                first_seen TEXT DEFAULT CURRENT_TIMESTAMP,
                last_seen TEXT DEFAULT CURRENT_TIMESTAMP
            );

            CREATE INDEX IF NOT EXISTS idx_outcomes_category
                ON optimization_outcomes(kernel_category, bottleneck_type);
            CREATE INDEX IF NOT EXISTS idx_commandment_category
                ON commandment_patterns(kernel_category);
        """)
        self._conn.commit()

    def _migrate_columns(self):
        """Add columns that may be missing from older DB schemas."""
        for col, col_type, default in [
            ("optimization_technique", "TEXT", "''"),
            ("kernel_language", "TEXT", "''"),
            ("patch_snippet", "TEXT", "''"),
        ]:
            try:
                self._conn.execute(
                    f"ALTER TABLE optimization_outcomes ADD COLUMN {col} {col_type} DEFAULT {default}"
                )
                self._conn.commit()
            except sqlite3.OperationalError:
                pass

    def record_outcome(
        self,
        kernel_type: str = "triton",
        kernel_category: str = "unknown",
        bottleneck_type: str = "unknown",
        gpu_architecture: str = "unknown",
        strategy_name: str = "",
        speedup_achieved: float = 1.0,
        success: bool = False,
        failure_reason: str | None = None,
        cost_dollars: float = 0.0,
        steps_taken: int = 0,
        commandment_worked: bool = False,
        profiling_metrics: dict | None = None,
        kernel_signature_hash: str = "",
        patch_snippet: str = "",
    ) -> int:
        """Record an optimization outcome with optional code snippet."""
        cursor = self._conn.execute(
            """INSERT INTO optimization_outcomes
               (kernel_type, kernel_category, bottleneck_type, gpu_architecture,
                strategy_name, speedup_achieved, success, failure_reason,
                cost_dollars, steps_taken, commandment_worked, profiling_metrics,
                kernel_signature_hash, patch_snippet)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                kernel_type, kernel_category, bottleneck_type, gpu_architecture,
                strategy_name, speedup_achieved, int(success), failure_reason,
                cost_dollars, steps_taken, int(commandment_worked),
                json.dumps(profiling_metrics or {}),
                kernel_signature_hash,
                patch_snippet[:2000],
            ),
        )
        self._conn.commit()
        return cursor.lastrowid

    def record_commandment_pattern(
        self,
        kernel_category: str,
        template_name: str,
        commandment_content: str,
        success: bool = True,
    ):
        """Record a COMMANDMENT pattern and its success/failure."""
        existing = self._conn.execute(
            "SELECT id, success_count, failure_count FROM commandment_patterns WHERE kernel_category=? AND template_name=?",
            (kernel_category, template_name),
        ).fetchone()

        if existing:
            if success:
                self._conn.execute(
                    "UPDATE commandment_patterns SET success_count=?, last_used=CURRENT_TIMESTAMP WHERE id=?",
                    (existing["success_count"] + 1, existing["id"]),
                )
            else:
                self._conn.execute(
                    "UPDATE commandment_patterns SET failure_count=?, last_used=CURRENT_TIMESTAMP WHERE id=?",
                    (existing["failure_count"] + 1, existing["id"]),
                )
        else:
            self._conn.execute(
                """INSERT INTO commandment_patterns
                   (kernel_category, template_name, commandment_content, success_count, failure_count)
                   VALUES (?, ?, ?, ?, ?)""",
                (kernel_category, template_name, commandment_content,
                 1 if success else 0, 0 if success else 1),
            )
        self._conn.commit()

    def record_pitfall(self, kernel_category: str, pitfall_type: str, description: str):
        """Record a known pitfall for a kernel category."""
        existing = self._conn.execute(
            "SELECT id, occurrences FROM known_pitfalls WHERE kernel_category=? AND pitfall_type=? AND description=?",
            (kernel_category, pitfall_type, description),
        ).fetchone()

        if existing:
            self._conn.execute(
                "UPDATE known_pitfalls SET occurrences=?, last_seen=CURRENT_TIMESTAMP WHERE id=?",
                (existing["occurrences"] + 1, existing["id"]),
            )
        else:
            self._conn.execute(
                "INSERT INTO known_pitfalls (kernel_category, pitfall_type, description) VALUES (?, ?, ?)",
                (kernel_category, pitfall_type, description),
            )
        self._conn.commit()

    def query_outcomes(
        self,
        kernel_category: str | None = None,
        bottleneck_type: str | None = None,
        limit: int = 20,
    ) -> list[dict]:
        """Query past optimization outcomes."""
        conditions = []
        params = []
        if kernel_category:
            conditions.append("kernel_category = ?")
            params.append(kernel_category)
        if bottleneck_type:
            conditions.append("bottleneck_type = ?")
            params.append(bottleneck_type)

        where = f"WHERE {' AND '.join(conditions)}" if conditions else ""
        rows = self._conn.execute(
            f"SELECT * FROM optimization_outcomes {where} ORDER BY timestamp DESC LIMIT ?",
            params + [limit],
        ).fetchall()
        return [dict(r) for r in rows]

    def query_strategy_effectiveness(
        self,
        kernel_category: str | None = None,
        bottleneck_type: str | None = None,
    ) -> list[dict]:
        """Get strategy effectiveness stats (success rate, avg speedup)."""
        conditions = []
        params = []
        if kernel_category:
            conditions.append("kernel_category = ?")
            params.append(kernel_category)
        if bottleneck_type:
            conditions.append("bottleneck_type = ?")
            params.append(bottleneck_type)

        where = f"WHERE {' AND '.join(conditions)}" if conditions else ""
        rows = self._conn.execute(
            f"""SELECT strategy_name,
                       COUNT(*) as total_attempts,
                       SUM(success) as successes,
                       AVG(speedup_achieved) as avg_speedup,
                       MAX(speedup_achieved) as max_speedup,
                       GROUP_CONCAT(DISTINCT failure_reason) as failure_reasons
                FROM optimization_outcomes {where}
                GROUP BY strategy_name
                ORDER BY AVG(speedup_achieved) DESC""",
            params,
        ).fetchall()
        return [dict(r) for r in rows]

    def query_pitfalls(self, kernel_category: str) -> list[dict]:
        """Get known pitfalls for a kernel category."""
        rows = self._conn.execute(
            "SELECT * FROM known_pitfalls WHERE kernel_category=? ORDER BY occurrences DESC",
            (kernel_category,),
        ).fetchall()
        return [dict(r) for r in rows]

    def get_best_commandment(self, kernel_category: str) -> str | None:
        """Get the most successful COMMANDMENT template for a kernel category."""
        row = self._conn.execute(
            """SELECT commandment_content FROM commandment_patterns
               WHERE kernel_category=? AND success_count > 0
               ORDER BY (success_count * 1.0 / (success_count + failure_count + 1)) DESC
               LIMIT 1""",
            (kernel_category,),
        ).fetchone()
        return row["commandment_content"] if row else None

    def query_by_bottleneck_similarity(
        self,
        bottleneck_type: str,
        profiling_metrics: dict | None = None,
        limit: int = 5,
    ) -> list[dict]:
        """Retrieve outcomes from ANY kernel category with similar bottleneck profile.

        This enables cross-kernel transfer: a rope kernel and a gemm kernel
        with the same LDS bottleneck share optimization insights.

        Inspired by KernelBlaster (2602.14293): Persistent CUDA Knowledge Base.
        """
        rows = self._conn.execute(
            """SELECT * FROM optimization_outcomes
               WHERE bottleneck_type = ? AND success = 1
               ORDER BY speedup_achieved DESC LIMIT ?""",
            (bottleneck_type, limit),
        ).fetchall()
        return [dict(r) for r in rows]

    def format_memory_context(
        self,
        kernel_category: str | None = None,
        bottleneck_type: str | None = None,
        profiling_metrics: dict | None = None,
    ) -> str:
        """Format memory context with dual-retrieval for injection into agent task prompt.

        Dual retrieval (KernelBlaster + SMITH-inspired):
        1. Category match: retrieve outcomes for the same kernel category
        2. Bottleneck similarity: retrieve outcomes from ANY category with similar bottleneck
        """
        # Retrieval 1: Category match
        outcomes = self.query_outcomes(kernel_category, bottleneck_type, limit=50)

        # Retrieval 2: Bottleneck similarity (cross-kernel transfer)
        cross_kernel = []
        if bottleneck_type:
            cross_kernel = self.query_by_bottleneck_similarity(bottleneck_type, profiling_metrics)
            cross_kernel = [o for o in cross_kernel if o.get("kernel_category") != kernel_category]

        if not outcomes and not cross_kernel:
            return ""

        strategies = self.query_strategy_effectiveness(kernel_category, bottleneck_type)
        pitfalls = self.query_pitfalls(kernel_category) if kernel_category else []

        total = len(outcomes) + len(cross_kernel)
        lines = [f"--- Optimization Memory (from {total} past runs) ---"]
        if kernel_category:
            lines.append(f"Kernel category: {kernel_category}")

        successful = [o for o in outcomes if o["success"]]
        if successful:
            avg_speedup = sum(o["speedup_achieved"] for o in successful) / len(successful)
            lines.append(f"Successful optimizations: {len(successful)}/{len(outcomes)} (avg speedup: {avg_speedup:.2f}x)")

        if strategies:
            worked = [s for s in strategies if s["successes"] and s["successes"] > 0]
            failed = [s for s in strategies if not s["successes"] or s["successes"] == 0]

            if worked:
                lines.append("\nStrategies that worked:")
                for s in worked[:5]:
                    rate = s["successes"] / s["total_attempts"] if s["total_attempts"] else 0
                    lines.append(
                        f"  - {s['strategy_name']}: {s['successes']}/{s['total_attempts']} success, "
                        f"avg {s['avg_speedup']:.2f}x speedup"
                    )

            if failed:
                lines.append("\nStrategies that failed:")
                for s in failed[:5]:
                    reason = s.get("failure_reasons", "unknown")
                    lines.append(f"  - {s['strategy_name']}: 0/{s['total_attempts']} success ({reason})")

        # Show code snippets from best-performing past runs (AccelOpt-inspired)
        best_with_snippet = sorted(
            [o for o in successful if o.get("patch_snippet")],
            key=lambda o: o["speedup_achieved"], reverse=True,
        )
        if best_with_snippet:
            top = best_with_snippet[0]
            lines.append(f"\nBest code change that worked ({top['speedup_achieved']:.2f}x, "
                         f"strategy: {top.get('strategy_name', '?')}):")
            lines.append("```diff")
            snippet = top["patch_snippet"]
            if len(snippet) > 500:
                snippet = snippet[:500] + "\n... (truncated)"
            lines.append(snippet)
            lines.append("```")

        if pitfalls:
            lines.append("\nKnown pitfalls:")
            for p in pitfalls[:5]:
                lines.append(f"  - [{p['pitfall_type']}] {p['description']} (seen {p['occurrences']}x)")

        # Cross-kernel transfer section (KernelBlaster-inspired)
        if cross_kernel:
            lines.append(f"\nCross-kernel insights (similar bottleneck: {bottleneck_type}):")
            for o in cross_kernel[:3]:
                cat = o.get("kernel_category", "?")
                sp = o.get("speedup_achieved", 0)
                strat = o.get("strategy_name", "?")
                lines.append(f"  - {cat} ({strat}): {sp:.2f}x speedup with same bottleneck type")

        # Profile-metric similarity search (cosine similarity on metric vectors)
        if profiling_metrics:
            try:
                from minisweagent.memory.profile_similarity import find_similar_profiles, format_similar_profiles
                all_outcomes = self.query_outcomes(limit=100)
                similar = find_similar_profiles(profiling_metrics, all_outcomes, top_k=3)
                sim_block = format_similar_profiles(similar, exclude_category=kernel_category)
                if sim_block:
                    lines.append(sim_block)
            except Exception:
                pass

        lines.append("---")
        return "\n".join(lines)

    def get_stats(self) -> dict:
        """Get overall memory statistics."""
        total = self._conn.execute("SELECT COUNT(*) FROM optimization_outcomes").fetchone()[0]
        successful = self._conn.execute("SELECT COUNT(*) FROM optimization_outcomes WHERE success=1").fetchone()[0]
        categories = self._conn.execute("SELECT DISTINCT kernel_category FROM optimization_outcomes").fetchall()
        patterns = self._conn.execute("SELECT COUNT(*) FROM commandment_patterns").fetchone()[0]
        pitfalls = self._conn.execute("SELECT COUNT(*) FROM known_pitfalls").fetchone()[0]
        return {
            "total_outcomes": total,
            "successful_outcomes": successful,
            "kernel_categories": [r[0] for r in categories],
            "commandment_patterns": patterns,
            "known_pitfalls": pitfalls,
        }

    def close(self):
        self._conn.close()


def classify_kernel_category(kernel_path: str | Path) -> str:
    """Classify a kernel file into a category based on its name and content."""
    path = Path(kernel_path)
    name = path.stem.lower()
    content = path.read_text().lower() if path.exists() else ""

    if "gemm" in name or "matmul" in name or "mm_kernel" in content:
        return "gemm"
    if "rope" in name or "rotary" in content:
        return "rope"
    if "topk" in name or "top_k" in name:
        return "topk"
    if "attention" in name or "attn" in name or "nsa" in name:
        return "attention"
    if "norm" in name or "rmsnorm" in content or "layernorm" in content:
        return "normalization"
    if "ff_" in name or "feedforward" in name or "mlp" in name:
        return "feedforward"
    if "fused" in name:
        return "fused"
    if "moe" in name or "mixture" in content:
        return "moe"
    if "softmax" in name:
        return "softmax"
    if "conv" in name:
        return "convolution"
    return "unknown"


def extract_outcome_from_agent_log(log_path: str | Path) -> dict | None:
    """Parse an agent log to extract the optimization outcome for memory storage.

    Returns a dict suitable for record_outcome(), or None if parsing fails.
    """
    import re

    path = Path(log_path)
    if not path.exists():
        return None

    text = path.read_text()

    # Extract steps and cost
    steps_matches = re.findall(r'step (\d+) \(\$([0-9.]+)\)', text)
    if steps_matches:
        steps, cost = int(steps_matches[-1][0]), float(steps_matches[-1][1])
    else:
        steps, cost = 0, 0.0

    # Extract speedup geomean
    speedups = re.findall(r'Speedup \(geomean\):\s+(\d+\.\d+)x', text)
    speedup = float(speedups[-1]) if speedups else 1.0

    # Check COMMANDMENT success
    commandment_worked = "COMMANDMENT.md validation: OK" in text or "COMMANDMENT auto-generated" in text

    # Check OE result
    oe_best = re.findall(r'best speedup: (\d+\.\d+)x', text)
    oe_speedup = float(oe_best[-1]) if oe_best else 0.0

    # Detect bottleneck from profiling output
    bottleneck = "unknown"
    bn_matches = re.findall(r'"bottleneck":\s*"(\w+)"', text)
    if bn_matches:
        bottleneck = bn_matches[0]

    # Determine success
    success = speedup > 1.0 or oe_speedup > 1.0

    # Failure reason
    failure_reason = None
    if not success:
        if "rocBLAS" in text or "F.linear" in text:
            failure_reason = "rocBLAS/cuBLAS is fundamentally faster for this operation"
        elif "BrokenPipeError" in text:
            failure_reason = "OpenEvolve crashed with BrokenPipeError"
        elif not commandment_worked:
            failure_reason = "COMMANDMENT.md creation failed"

    return {
        "speedup_achieved": max(speedup, oe_speedup),
        "success": success,
        "failure_reason": failure_reason,
        "cost_dollars": cost,
        "steps_taken": steps,
        "commandment_worked": commandment_worked,
        "bottleneck_type": bottleneck,
    }
