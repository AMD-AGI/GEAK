"""Working Memory for GEAK Agent (Within-Session).

Maintains a compact, structured state that is injected into every LLM call,
preventing context saturation (B4), agent spinning (B2), and providing
real-time feedback (B6).

Components:
1. Session State Tracker (~300 tokens): phase, strategies tried, best speedup
2. Insight Buffer (~200 tokens): rolling window of 5 WIN/FAIL/OK insights
3. Progress Monitor (~100 tokens): speedup trajectory + early-stop signals
4. Cost/Step Budget (~100 tokens): hard limits with graceful degradation

Total budget: ~800 tokens hard cap.

Inspired by:
- CogMem (2512.14118): Focus of Attention mechanism
- MEM1 (2506.15841): Constant-memory via reasoning-driven consolidation
- Colleague's insight buffer: zero-cost WIN/FAIL/OK extraction
"""

from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass, field
from typing import Any


MAX_WORKING_MEMORY_TOKENS = 800
MAX_INSIGHTS = 5


@dataclass
class Insight:
    """A single causal insight extracted from a tool result."""
    step: int
    tag: str  # WIN, FAIL, OK, WARN
    message: str
    timestamp: float = field(default_factory=time.time)

    def format(self) -> str:
        return f"[{self.tag}] step {self.step}: {self.message}"


@dataclass
class WorkingMemory:
    """Compact within-session memory injected into every LLM call."""

    # Session state
    phase: str = "discovery"  # discovery, profiling, strategy, optimization, reporting
    current_step: int = 0
    current_cost: float = 0.0
    best_speedup: float = 0.0
    best_speedup_step: int = 0
    strategies_tried: list[str] = field(default_factory=list)
    strategies_failed: list[str] = field(default_factory=list)
    current_action: str = ""
    kernel_category: str = "unknown"

    # Insight buffer (rolling window)
    insights: list[Insight] = field(default_factory=list)

    # Budget
    max_cost: float = 0.50
    max_steps: int = 100

    # Progress tracking
    speedup_history: list[tuple[int, float]] = field(default_factory=list)
    steps_since_improvement: int = 0
    baseline_latency_ms: float = 0.0
    best_latency_ms: float = 0.0
    bottleneck_type: str = ""
    latency_history: list[float] = field(default_factory=list)
    tuning_steps: int = 0
    algo_steps: int = 0

    def update_step(self, step: int, cost: float):
        """Called after each agent step."""
        self.current_step = step
        self.current_cost = cost

    def update_speedup(self, speedup: float):
        """Record a new speedup measurement."""
        self.speedup_history.append((self.current_step, speedup))
        if speedup > self.best_speedup:
            self.best_speedup = speedup
            self.best_speedup_step = self.current_step
            self.steps_since_improvement = 0
        else:
            self.steps_since_improvement += 1

    def update_latency(self, latency_ms: float):
        """Record a benchmark latency and compute speedup vs baseline."""
        self.latency_history.append(latency_ms)
        if self.baseline_latency_ms <= 0:
            self.baseline_latency_ms = latency_ms
        if self.best_latency_ms <= 0 or latency_ms < self.best_latency_ms:
            self.best_latency_ms = latency_ms
        if self.baseline_latency_ms > 0 and latency_ms > 0:
            speedup = self.baseline_latency_ms / latency_ms
            self.update_speedup(speedup)

    def is_diminishing_returns(self) -> bool:
        """Check if last 3 latencies are within 1% of each other."""
        if len(self.latency_history) < 3:
            return False
        last3 = self.latency_history[-3:]
        avg = sum(last3) / 3
        return all(abs(v - avg) / avg < 0.01 for v in last3) if avg > 0 else False

    def add_insight(self, tag: str, message: str):
        """Add a causal insight. Maintains rolling window of MAX_INSIGHTS."""
        self.insights.append(Insight(
            step=self.current_step, tag=tag, message=message[:120],
        ))
        if len(self.insights) > MAX_INSIGHTS:
            self.insights = self.insights[-MAX_INSIGHTS:]

    def record_strategy(self, name: str, success: bool):
        """Record a strategy attempt."""
        if name not in self.strategies_tried:
            self.strategies_tried.append(name)
        if not success and name not in self.strategies_failed:
            self.strategies_failed.append(name)

    def get_progress_signal(self) -> str:
        """Get progress/early-stop signal."""
        if self.steps_since_improvement > 20 and self.best_speedup > 0:
            return f"EARLY_STOP: No improvement for {self.steps_since_improvement} steps. Best={self.best_speedup:.2f}x at step {self.best_speedup_step}. Submit now."
        if self.steps_since_improvement > 10 and self.best_speedup > 0:
            return f"STALLED: No improvement for {self.steps_since_improvement} steps. Best={self.best_speedup:.2f}x. Consider submitting."
        if len(self.speedup_history) >= 2:
            recent = self.speedup_history[-1][1]
            prev = self.speedup_history[-2][1]
            if recent > prev:
                return f"PROGRESS: Speedup improving ({prev:.2f}x -> {recent:.2f}x)"
        return ""

    def get_budget_signal(self) -> str:
        """Get cost/step budget signal."""
        cost_pct = self.current_cost / self.max_cost if self.max_cost > 0 else 0
        step_pct = self.current_step / self.max_steps if self.max_steps > 0 else 0
        pct = max(cost_pct, step_pct)

        if pct >= 0.95:
            return f"BUDGET_FORCE: ${self.current_cost:.2f}/${self.max_cost:.2f}, step {self.current_step}/{self.max_steps}. MUST submit immediately."
        if pct >= 0.85:
            return f"BUDGET_CRITICAL: ${self.current_cost:.2f}/${self.max_cost:.2f}, step {self.current_step}/{self.max_steps}. Wrap up and submit best result."
        if pct >= 0.70:
            return f"BUDGET_WARN: ${self.current_cost:.2f}/${self.max_cost:.2f}, step {self.current_step}/{self.max_steps}. ~{int((1-pct)*self.max_steps)} steps remaining."
        return ""

    def format_for_injection(self) -> str:
        """Format working memory for injection into LLM prompt (~800 tokens)."""
        parts = []

        parts.append(f"--- Working Memory (step {self.current_step}, ${self.current_cost:.2f}) ---")
        parts.append(
            "PRIORITY: (1) Algorithmic kernel body rewrites (different algorithm, fused ops, "
            "eliminate reshape/flip, split kernels) > (2) Operation fusion > "
            "(3) Memory/compute restructuring > (4) Parameter tuning (ONLY after exhausting 1-3)."
        )

        best_str = f"{self.best_speedup:.2f}x"
        if self.best_latency_ms > 0:
            best_str += f" ({self.best_latency_ms:.4f}ms vs baseline {self.baseline_latency_ms:.4f}ms)"
        parts.append(f"Kernel: {self.kernel_category} | Best: {best_str}")
        if self.strategies_tried:
            parts.append(f"Tried: {', '.join(self.strategies_tried[-5:])}")
        if self.strategies_failed:
            parts.append(f"Failed: {', '.join(self.strategies_failed[-3:])}")

        if self.tuning_steps >= 4 and self.steps_since_improvement >= 3:
            parts.append(
                "SWITCH: Parameter tuning saturated. Try algorithmic kernel restructuring "
                "(split kernel variants, eliminate tl.reshape/tl.flip, fuse adjacent operations)."
            )
        elif self.is_diminishing_returns():
            parts.append(
                "DIMINISHING: Last 3 results within 1% of each other. "
                "Current approach exhausted. Try these (in order): "
                "(1) @triton.autotune with multiple BLOCK_S/num_warps configs so Triton picks the best per input shape, "
                "(2) Shape-category dispatch: write 2-3 kernel variants (small S vs large S, or D=64 vs D=128) "
                "and select in the Python wrapper based on input shape, "
                "(3) A fundamentally different algorithm for the computation."
            )

        if self.bottleneck_type and self.tuning_steps >= 2:
            _bn_hint = {
                "balanced": (
                    "Bottleneck is balanced -- parameter tuning won't help. Focus on algorithmic changes. "
                    "If improvement varies across input shapes, use @triton.autotune or shape-category dispatch."
                ),
                "memory-bound": "Bottleneck is memory -- try vectorized loads, LDS staging, or fuse ops to reduce traffic.",
                "compute-bound": "Bottleneck is compute -- try MFMA instructions, reduce instruction count, or fuse ops.",
                "latency-bound": (
                    "Bottleneck is latency -- increase work per kernel, fuse with adjacent kernels. "
                    "If geomean is low despite some shapes improving, use @triton.autotune to pick optimal config per shape."
                ),
            }
            hint = _bn_hint.get(self.bottleneck_type)
            if hint:
                parts.append(hint)

        # Insight buffer (~200 tokens)
        if self.insights:
            parts.append("")
            parts.append("Recent insights:")
            for ins in self.insights[-MAX_INSIGHTS:]:
                parts.append(f"  {ins.format()}")

        # Progress signal (~100 tokens)
        progress = self.get_progress_signal()
        if progress:
            parts.append("")
            parts.append(progress)

        # Budget signal (~100 tokens)
        budget = self.get_budget_signal()
        if budget:
            parts.append(budget)

        parts.append("---")
        return "\n".join(parts)


def classify_change(text: str) -> str:
    """Classify a code change as algorithmic, fusion, tuning, or wrapper."""
    algo = [r'def \w+_kernel', r'split.*kernel', r'tl\.reshape|tl\.flip', r'direct.index',
            r'half.dim', r'different.*algorithm', r'rewrite', r'restructur']
    fusion = [r'fuse|fusion', r'fused_', r'merge.*kernel', r'combine.*ops']
    tuning = [r'BLOCK_S\s*=', r'num_warps\s*=', r'num_stages\s*=', r'@triton\.autotune',
              r'waves_per_eu', r'BLOCK_SIZE\s*=']
    for p in algo:
        if re.search(p, text, re.IGNORECASE):
            return "algorithmic"
    for p in fusion:
        if re.search(p, text, re.IGNORECASE):
            return "fusion"
    for p in tuning:
        if re.search(p, text):
            return "tuning"
    return "wrapper"


def summarize_change(text: str) -> str:
    """Extract a brief ALGO/TUNE/FUSION prefix from code content."""
    indicators = [
        (r'tl\.flip|tl\.reshape', 'ALGO(reshape/flip change)'),
        (r'def \w+_kernel\w*\(', 'ALGO(new/split kernel)'),
        (r'half.dim|BLOCK_D_HALF|x1.*x2', 'ALGO(half-dim loads)'),
        (r'fuse|fusion|fused_', 'FUSION(fused ops)'),
        (r'scale_inv|precompute', 'ALGO(precompute)'),
        (r'contiguous|stride.*removed', 'ALGO(memory layout)'),
        (r'vectori|float[24]', 'ALGO(vectorized access)'),
        (r'BLOCK_S\s*=\s*(\d+)', 'TUNE(BLOCK_S={})'),
        (r'num_warps\s*=\s*(\d+)', 'TUNE(num_warps={})'),
        (r'num_stages\s*=\s*(\d+)', 'TUNE(num_stages={})'),
        (r'@triton\.autotune', 'TUNE(autotune)'),
    ]
    for pat, desc in indicators:
        m = re.search(pat, text, re.IGNORECASE)
        if m:
            return desc.format(m.group(1)) if '{}' in desc and m.lastindex else desc
    return "EDIT"


def extract_strategy_from_edit(edit_content: str) -> str | None:
    """Extract optimization strategy keywords from a kernel edit."""
    return summarize_change(edit_content) if edit_content else None


def extract_insight_from_tool_result(tool_name: str, output: str, returncode: int) -> Insight | None:
    """Extract a causal insight from a tool call result without an LLM call.

    Uses regex/keyword matching for zero-cost extraction.
    """
    if not output:
        return None

    output_lower = output.lower()

    # Profiling results
    if "bottleneck" in output_lower and ("memory" in output_lower or "compute" in output_lower or "latency" in output_lower or "lds" in output_lower):
        bn_match = re.search(r'"bottleneck":\s*"(\w+)"', output)
        if bn_match:
            return Insight(step=0, tag="OK", message=f"Profiling: bottleneck={bn_match.group(1)}")

    # GEAK benchmark latency (most precise kernel metric)
    latency_match = re.search(r'GEAK_RESULT_LATENCY_MS=(\d+\.\d+)', output)
    if latency_match:
        lat = float(latency_match.group(1))
        return Insight(step=0, tag="OK", message=f"Benchmark latency: {lat:.4f}ms")

    # Speedup results
    speedup_match = re.search(r'Speedup \(geomean\):\s+(\d+\.\d+)x', output)
    if speedup_match:
        sp = float(speedup_match.group(1))
        tag = "WIN" if sp > 1.0 else "FAIL" if sp < 0.5 else "OK"
        return Insight(step=0, tag=tag, message=f"Speedup geomean: {sp:.2f}x")

    # Correctness
    if "all pass" in output_lower or "all_pass" in output_lower:
        return Insight(step=0, tag="OK", message="Correctness: ALL PASS")
    if "fail" in output_lower and returncode != 0:
        fail_match = re.search(r'(FAIL|Error|failed).*?$', output, re.MULTILINE)
        msg = fail_match.group(0)[:80] if fail_match else "Test failed"
        return Insight(step=0, tag="FAIL", message=msg)

    # COMMANDMENT validation
    if "commandment.md validation: ok" in output_lower:
        return Insight(step=0, tag="OK", message="COMMANDMENT validated successfully")
    if "commandment.md validation error" in output_lower:
        return Insight(step=0, tag="FAIL", message="COMMANDMENT validation failed")

    # OpenEvolve progress
    oe_match = re.search(r'best speedup: (\d+\.\d+)x', output)
    if oe_match:
        sp = float(oe_match.group(1))
        tag = "WIN" if sp > 1.0 else "OK"
        return Insight(step=0, tag=tag, message=f"OpenEvolve best: {sp:.2f}x")

    # BrokenPipeError (common OE failure)
    if "brokenpipeerror" in output_lower:
        return Insight(step=0, tag="FAIL", message="OpenEvolve BrokenPipeError -- process crash")

    # Generic error
    if returncode != 0 and len(output) > 10:
        last_line = output.strip().splitlines()[-1][:80] if output.strip() else "unknown error"
        return Insight(step=0, tag="FAIL", message=f"Exit {returncode}: {last_line}")

    return None
