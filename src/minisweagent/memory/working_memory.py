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

from minisweagent.memory.working_notebook import WorkingNotebook, summarize_working_notebook


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
    profiler_diagnosis: str = ""
    noise_floor_pct: float = 0.0
    consecutive_same_category: int = 0
    last_change_category: str = ""
    notebook_dir: str | None = None
    notebook_writer_id: str = "default"
    best_strategy: str = ""
    best_change_category: str = ""
    pending_strategy: str = ""
    pending_change_category: str = ""
    _notebook: WorkingNotebook | None = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        if self.notebook_dir:
            try:
                self._notebook = WorkingNotebook(
                    self.notebook_dir,
                    writer_id=self.notebook_writer_id or "default",
                )
            except Exception:
                self._notebook = None

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

    def sync_notebook_baseline(self) -> None:
        """Persist the current baseline metadata into the working notebook."""
        if not self._notebook:
            return
        self._notebook.record_baseline(
            baseline_latency_ms=self.baseline_latency_ms or None,
            bottleneck_type=self.bottleneck_type or None,
            kernel_category=self.kernel_category or None,
        )

    def remember_pending_change(self, strategy: str, change_category: str) -> None:
        """Remember the last concrete edit so benchmark results can be linked to it."""
        self.pending_strategy = strategy
        self.pending_change_category = change_category
        if self._notebook:
            self._notebook.record_attempt(
                strategy=strategy,
                change_category=change_category,
                step=self.current_step,
            )

    def note_tool_result(
        self,
        output: str,
        returncode: int,
        *,
        tag: str | None = None,
        message: str | None = None,
    ) -> None:
        """Persist a tool result into the file-backed notebook."""
        if not self._notebook or not output:
            return
        self._notebook.record_result(
            output=output,
            returncode=returncode,
            strategy=self.pending_strategy or None,
            change_category=self.pending_change_category or None,
            tag=tag,
            message=message,
            step=self.current_step,
        )

        overall = re.search(r"Overall:\s*([0-9]+(?:\.[0-9]+)?)x", output, re.IGNORECASE)
        if overall:
            speedup = float(overall.group(1))
        else:
            # Fallback: parse GEAK_RESULT_GEOMEAN_SPEEDUP or GEAK_RESULT_SPEEDUP when Overall: absent
            # (e.g. moe harness before fix, or harnesses that only output speedup)
            fallback = re.search(
                r"GEAK_RESULT_(?:GEOMEAN_)?SPEEDUP=([0-9]+(?:\.[0-9]+)?)",
                output,
                re.IGNORECASE,
            )
            if fallback:
                speedup = float(fallback.group(1))
            else:
                speedup = None
        if speedup is not None and speedup > self.best_speedup:
            self.best_strategy = self.pending_strategy
            self.best_change_category = self.pending_change_category
            self.update_speedup(speedup)

        if "Patch saved:" in output or "Test status:" in output:
            self.pending_strategy = ""
            self.pending_change_category = ""

    def record_round_evaluation(self, round_eval: dict[str, Any]) -> None:
        """Store verified round-level evidence into the working notebook."""
        full_benchmark = round_eval.get("full_benchmark", {}) if isinstance(round_eval, dict) else {}
        verified_speedup = (
            full_benchmark.get("verified_speedup")
            if isinstance(full_benchmark, dict)
            else None
        )
        if verified_speedup:
            self.update_speedup(float(verified_speedup))
            self.best_strategy = str(round_eval.get("best_task") or self.best_strategy)
        candidate_ms = (
            full_benchmark.get("candidate_ms")
            if isinstance(full_benchmark, dict)
            else None
        )
        if candidate_ms:
            candidate_val = float(candidate_ms)
            if self.best_latency_ms <= 0 or candidate_val < self.best_latency_ms:
                self.best_latency_ms = candidate_val
        if not self._notebook:
            return
        self._notebook.record_round_evaluation(
            round_num=int(round_eval.get("round", 0) or 0),
            best_task=round_eval.get("best_task"),
            verified_speedup=verified_speedup,
            baseline_ms=(
                full_benchmark.get("baseline_ms")
                if isinstance(full_benchmark, dict)
                else None
            ),
            candidate_ms=candidate_ms,
            per_shape_speedups=round_eval.get("per_shape_speedups"),
        )

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

    def record_change_category(self, category: str):
        """Track consecutive same-category changes for diversity enforcement."""
        if category == self.last_change_category:
            self.consecutive_same_category += 1
        else:
            self.consecutive_same_category = 1
            self.last_change_category = category
        if category == "tuning":
            self.tuning_steps += 1
            self.algo_steps = 0
        elif category in ("algorithmic", "fusion"):
            self.algo_steps += 1
            self.tuning_steps = 0

    def format_for_injection(self) -> str:
        """Format working memory for injection into LLM prompt."""
        parts = []

        parts.append(f"--- Working Memory (step {self.current_step}) ---")
        parts.append(
            "PRIORITY: (1) Algorithmic kernel rewrites > (2) Operation fusion > "
            "(3) Memory restructuring > (4) Parameter tuning > "
            "(5) Dispatch-path / wrapper changes LAST."
        )

        best_str = f"{self.best_speedup:.2f}x"
        if self.best_latency_ms > 0:
            best_str += f" ({self.best_latency_ms:.4f}ms vs baseline {self.baseline_latency_ms:.4f}ms)"
        parts.append(f"Kernel: {self.kernel_category} | Best: {best_str}")
        if self.best_strategy:
            category_suffix = f" [{self.best_change_category}]" if self.best_change_category else ""
            parts.append(f"Best strategy so far: {self.best_strategy}{category_suffix}")
        if self.strategies_tried:
            parts.append(f"Tried: {', '.join(self.strategies_tried[-5:])}")
        if self.strategies_failed:
            parts.append(f"Failed: {', '.join(self.strategies_failed[-3:])}")

        # V2 Change 1: Profiler-to-architecture diagnosis (first 5 steps only)
        if self.profiler_diagnosis and self.current_step <= 5:
            parts.append("")
            parts.append(self.profiler_diagnosis)

        # V2 Change 6: Insight checkpoint at step 5
        if self.current_step == 5 and self.best_speedup <= 1.01:
            parts.append(
                "[STEP 5 CHECKPOINT] Have you identified the DOMINANT bottleneck? "
                "Re-read profile.json. Look for: algorithmic shortcuts, fusion opportunities, "
                "unnecessary memory copies (repeat_interleave), redundant ops, and only then "
                "dispatch-path mismatches or unfused external library calls."
            )

        # V2 Change 2: Hard ceiling detector (replaces soft diminishing returns)
        if self.is_diminishing_returns() and self.tuning_steps >= 3:
            parts.append(
                "[CEILING REACHED] Last 3 benchmarks within noise. "
                "STOP parameter tuning. OPTIONS: (1) SUBMIT best result now, "
                "(2) Try fundamentally different algorithm, "
                "(3) Try @triton.autotune with shape-specific configs."
            )
        elif self.is_diminishing_returns():
            parts.append(
                "[DIMINISHING] Last 3 results within 1%. Try a different approach: "
                "@triton.autotune, shape-specialized kernel variants, or a different algorithm."
            )

        # V2 Change 4: Approach diversity enforcer
        if self.consecutive_same_category >= 3:
            cat = self.last_change_category
            parts.append(
                f"[DIVERSITY REQUIRED] Last {self.consecutive_same_category} changes were all {cat.upper()}. "
                "You MUST try a different category: "
                + ("ALGORITHMIC or FUSION." if cat == "tuning" else "TUNING or MEMORY LAYOUT." if cat == "algorithmic" else "ALGORITHMIC or TUNING.")
            )

        # Bottleneck guidance (only when relevant)
        if self.bottleneck_type and self.tuning_steps >= 2:
            _bn_hint = {
                "balanced": "Bottleneck: balanced -- parameter tuning won't help. Focus on algorithmic changes or fusion; treat dispatch-path edits as a last resort.",
                "memory-bound": "Bottleneck: memory -- try vectorized loads, LDS staging, or fuse ops.",
                "compute-bound": "Bottleneck: compute -- try MFMA, reduce instructions, or fuse ops.",
                "latency-bound": "Bottleneck: latency -- increase work per kernel, fuse with adjacent kernels, or try @triton.autotune.",
            }
            hint = _bn_hint.get(self.bottleneck_type)
            if hint:
                parts.append(hint)

        # V2 Change 5: Benchmark noise awareness
        if self.noise_floor_pct > 0 and self.tuning_steps >= 2:
            parts.append(f"Noise floor: ±{self.noise_floor_pct:.1f}%. Changes below this are NOISE, not signal.")

        # Insight buffer
        if self.insights:
            parts.append("")
            parts.append("Recent insights:")
            for ins in self.insights[-MAX_INSIGHTS:]:
                parts.append(f"  {ins.format()}")

        notebook_summary = summarize_working_notebook(self.notebook_dir)
        if notebook_summary:
            parts.append("")
            parts.append(notebook_summary)

        # V2 Change 7: Early submission trigger
        if self.steps_since_improvement > 8 and self.current_step > self.max_steps * 0.4:
            parts.append(
                f"[SUBMIT NOW] Best={self.best_speedup:.2f}x at step {self.best_speedup_step}. "
                f"No improvement in {self.steps_since_improvement} steps. Submit and let next round try differently."
            )
        else:
            progress = self.get_progress_signal()
            if progress:
                parts.append(progress)

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


def _summarize_change(text: str) -> str:
    """Summarize a change from diff-like text in a backend-agnostic way."""
    if not text:
        return "EDIT"

    diff_lines: list[str] = []
    added_lines: list[str] = []
    for raw_line in text.splitlines():
        line = raw_line.rstrip()
        if line.startswith(("+++ ", "--- ")):
            continue
        if line.startswith("@@"):
            diff_lines.append(line)
            continue
        if line.startswith(("+", "-")) and len(line) > 1:
            body = line[1:].strip()
            diff_lines.append(body)
            if line.startswith("+"):
                added_lines.append(body)
            continue
        if (
            '"old_str"' in line
            or '"new_str"' in line
            or '"old_string"' in line
            or '"new_string"' in line
            or '"old_text"' in line
            or '"new_text"' in line
        ):
            diff_lines.append(line)
            if '"new_str"' in line or '"new_string"' in line or '"new_text"' in line:
                added_lines.append(line)

    haystack = "\n".join(diff_lines) if diff_lines else text
    haystack = haystack[:4000]
    added_haystack = "\n".join(added_lines)[:4000]
    search_spaces = [added_haystack, haystack] if added_haystack else [haystack]

    for space in search_spaces:
        specific_tuning = re.search(
            r'\b(num_warps|num_stages|waves_per_eu|items_per_thread|threads_per_block|warps_per_block|'
            r'BLOCK[_A-Z0-9]*|TILE[_A-Z0-9]*|GROUP[_A-Z0-9]*)\s*=\s*([0-9]+)\b',
            space,
            re.IGNORECASE,
        )
        if specific_tuning:
            return f"TUNE({specific_tuning.group(1)}={specific_tuning.group(2)})"

    indicators = [
        (r'@triton\.autotune|triton\.Config|autotun', 'TUNE(autotune/config)'),
        (
            r'\b(aiter|ck|tensile|rocblas|cublas|cutlass|flash_attention|'
            r'scaled_dot_product_attention|enable_gqa|dispatch)\b',
            'PATH(dispatch/backend)',
        ),
        (r'\b(fuse|fuses|fused|fusing|merge\w*|combine\w*|single[-_ ]pass)\b', 'FUSION(op merge)'),
        (
            r'\b(repeat_interleave|contiguous|reshape|view|expand|transpose|permute|stride|layout)\b',
            'ALGO(data layout)',
        ),
        (r'\b(vector\w*|float2|float4|half2|half4|int2|int4|packed|simd|mfma|wmma)\b', 'ALGO(vectorization)'),
        (r'\b(__shared__|shared memory|lds|smem|cache\w*|prefetch|register\w*|coalesc\w*)\b', 'ALGO(memory hierarchy)'),
        (r'__global__|__device__|@triton\.jit|template\s*<|def\s+\w+\(|struct\s+\w+|class\s+\w+', 'ALGO(new kernel/helper)'),
        (r'\b(reduce\w*|scan\w*|sort\w*|heap\w*|bitonic|radix|attention|matmul|gemm|tile\w*|split\w*)\b', 'ALGO(algorithm rewrite)'),
    ]
    for space in search_spaces:
        for pat, desc in indicators:
            if re.search(pat, space, re.IGNORECASE):
                return desc
    return "EDIT"


def summarize_change(text: str) -> str:
    """Backward-compatible public wrapper for change summarization."""
    return _summarize_change(text)


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
        change_desc = _summarize_change(output)
        if change_desc != "EDIT":
            return Insight(step=0, tag="OK", message=f"{change_desc} -> latency: {lat:.4f}ms")
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
