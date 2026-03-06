"""UCB1-based anti-fixation for strategy selection.

Prevents the agent from getting stuck on one strategy family by:
1. Tracking attempt counts and success rates per strategy
2. Computing UCB1 exploration bonus for under-tried strategies
3. Detecting stagnation (no improvement in N consecutive strategies)
4. Forcing strategy switches when stagnation is detected

UCB1 score = success_rate + C * sqrt(ln(total_attempts) / strategy_attempts)
Where C is the exploration constant (default sqrt(2)).

Reference:
  - Multi-armed bandit (UCB1) for strategy selection
  - CodeIt/HER for partial-success relabeling
"""

from __future__ import annotations

import math
from typing import Any


EXPLORATION_CONSTANT = math.sqrt(2)
STAGNATION_THRESHOLD = 3


class StrategyTracker:
    """Tracks strategy attempts, successes, and computes UCB1 scores."""

    def __init__(self):
        self._strategies: dict[str, dict[str, Any]] = {}
        self._total_attempts = 0
        self._recent_results: list[tuple[str, float]] = []

    def record_attempt(self, strategy: str, speedup: float, success: bool):
        if strategy not in self._strategies:
            self._strategies[strategy] = {
                "attempts": 0,
                "successes": 0,
                "total_speedup": 0.0,
                "best_speedup": 0.0,
            }
        s = self._strategies[strategy]
        s["attempts"] += 1
        s["total_speedup"] += speedup
        if success:
            s["successes"] += 1
        if speedup > s["best_speedup"]:
            s["best_speedup"] = speedup
        self._total_attempts += 1
        self._recent_results.append((strategy, speedup))

    def ucb1_score(self, strategy: str) -> float:
        """Compute UCB1 score. Higher = more worth exploring."""
        if strategy not in self._strategies:
            return float("inf")
        s = self._strategies[strategy]
        if s["attempts"] == 0:
            return float("inf")
        success_rate = s["successes"] / s["attempts"]
        exploration = EXPLORATION_CONSTANT * math.sqrt(
            math.log(max(self._total_attempts, 1)) / s["attempts"]
        )
        return success_rate + exploration

    def rank_strategies(self, candidates: list[str]) -> list[tuple[str, float]]:
        """Rank candidate strategies by UCB1 score (highest first)."""
        scored = [(s, self.ucb1_score(s)) for s in candidates]
        scored.sort(key=lambda x: -x[1])
        return scored

    def is_stagnating(self, window: int = STAGNATION_THRESHOLD) -> bool:
        """Detect stagnation: no speedup improvement in last N attempts."""
        if len(self._recent_results) < window:
            return False
        recent = self._recent_results[-window:]
        best_recent = max(sp for _, sp in recent)
        if len(self._recent_results) > window:
            older = self._recent_results[:-window]
            best_older = max(sp for _, sp in older)
            return best_recent <= best_older
        return all(sp <= 1.0 for _, sp in recent)

    def suggest_next_strategy(self, candidates: list[str]) -> str | None:
        """Suggest the best next strategy to try based on UCB1 + stagnation."""
        if not candidates:
            return None
        ranked = self.rank_strategies(candidates)
        if self.is_stagnating():
            untried = [s for s, _ in ranked if s not in self._strategies]
            if untried:
                return untried[0]
            least_tried = min(
                ranked,
                key=lambda x: self._strategies.get(x[0], {}).get("attempts", 0),
            )
            return least_tried[0]
        return ranked[0][0]

    def get_guidance_text(self) -> str:
        """Format strategy guidance for injection into the agent prompt."""
        if not self._strategies:
            return ""
        lines = ["## Strategy Exploration Guidance (UCB1)"]
        ranked = self.rank_strategies(list(self._strategies.keys()))
        for strat, score in ranked[:6]:
            s = self._strategies[strat]
            rate = s["successes"] / s["attempts"] if s["attempts"] > 0 else 0
            lines.append(
                f"- {strat}: {s['attempts']} attempts, "
                f"{rate:.0%} success, best={s['best_speedup']:.2f}x, "
                f"UCB1={score:.2f}"
            )
        if self.is_stagnating():
            lines.append("\nSTAGNATION DETECTED: Try a completely different strategy family.")
        return "\n".join(lines)

    def format_for_prompt(self) -> str:
        return self.get_guidance_text()
