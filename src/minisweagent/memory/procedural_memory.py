"""Procedural Memory: Multi-Faceted Distiller + Strategic Principles.

After each optimization run, distills raw outcomes into:
1. Success patterns: what specific changes led to speedup
2. Failure triggers: what caused failures and why
3. Comparative insights: what distinguishes success from failure
4. Abstract strategic principles: transferable rules for future kernels

Inspired by:
- ReMe (2512.10696): multi-faceted distillation
- EvolveR (2510.16079): offline self-distillation into reusable principles
"""

from __future__ import annotations

import json
import re
from pathlib import Path

from minisweagent import global_config_dir

PRINCIPLES_PATH = global_config_dir / "strategic_principles.json"


def distill_from_agent_log(log_path: str | Path, kernel_category: str) -> dict:
    """Extract multi-faceted insights from an agent log.

    Returns dict with:
      - success_patterns: list of what worked
      - failure_triggers: list of what failed and why
      - comparative_insights: list of what distinguishes success/failure
      - strategic_principle: one abstract transferable rule
    """
    path = Path(log_path)
    if not path.exists():
        return {}

    text = path.read_text()

    success_patterns = []
    failure_triggers = []
    comparative_insights = []

    # Extract speedup improvements
    speedups = re.findall(r'Speedup \(geomean\):\s+(\d+\.\d+)x', text)
    if speedups:
        best = max(float(s) for s in speedups)
        if best > 1.0:
            success_patterns.append(f"Achieved {best:.2f}x geomean speedup for {kernel_category}")

    # Extract COMMANDMENT outcomes
    if "COMMANDMENT.md validation: OK" in text or "COMMANDMENT auto-generated" in text:
        success_patterns.append("COMMANDMENT.md created successfully")
    elif "COMMANDMENT" in text and ("MISSING" in text or "validation error" in text.lower()):
        failure_triggers.append("COMMANDMENT.md creation failed -- use auto-generation")

    # Extract OE outcomes
    oe_bests = re.findall(r'best speedup: (\d+\.\d+)x', text)
    if oe_bests:
        oe_best = max(float(s) for s in oe_bests)
        if oe_best > 1.0:
            success_patterns.append(f"OpenEvolve achieved {oe_best:.2f}x best speedup")
        elif oe_best <= 1.0:
            failure_triggers.append("OpenEvolve returned 1.0x -- no improvement found")

    # Extract BrokenPipeError
    if "BrokenPipeError" in text:
        failure_triggers.append("OpenEvolve crashed with BrokenPipeError")

    # Extract rocBLAS/vendor BLAS insights
    if "rocBLAS" in text or "F.linear" in text:
        failure_triggers.append("Triton kernel competes against vendor BLAS (rocBLAS/cuBLAS)")
        comparative_insights.append(
            "Vendor BLAS is faster for standard operations; "
            "focus on fused/custom operations where vendor has no equivalent"
        )

    # Extract bottleneck info
    bn_matches = re.findall(r'"bottleneck":\s*"(\w+)"', text)
    if bn_matches:
        bottleneck = bn_matches[0]
        success_patterns.append(f"Bottleneck identified: {bottleneck}")

    # Generate strategic principle
    principle = _generate_principle(kernel_category, success_patterns, failure_triggers)

    return {
        "kernel_category": kernel_category,
        "success_patterns": success_patterns,
        "failure_triggers": failure_triggers,
        "comparative_insights": comparative_insights,
        "strategic_principle": principle,
    }


def _generate_principle(category: str, successes: list, failures: list) -> str:
    """Generate an abstract transferable principle from patterns."""
    if any("vendor BLAS" in f for f in failures):
        return (
            f"For {category}-like kernels: Triton cannot beat vendor BLAS for standard operations. "
            f"Focus optimization on fused/custom operations or quantization-specific paths."
        )
    if any("BrokenPipeError" in f for f in failures):
        return (
            f"For {category}-like kernels: OpenEvolve may crash due to process isolation issues. "
            f"Use manual optimization as fallback."
        )
    if any("COMMANDMENT" in f for f in failures):
        return (
            f"For {category}-like kernels: Always use auto-generated COMMANDMENT.md. "
            f"Manual creation is error-prone."
        )
    if successes and any("speedup" in s.lower() for s in successes):
        return (
            f"For {category}-like kernels: Profile first, then use OpenEvolve with auto-COMMANDMENT. "
            f"This category responds well to automated optimization."
        )
    return f"For {category}-like kernels: Profile to identify bottleneck, then apply targeted strategies."


class PrincipleStore:
    """Persistent store for distilled strategic principles."""

    def __init__(self, path: str | Path | None = None):
        self.path = Path(path) if path else PRINCIPLES_PATH
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._load()

    def _load(self):
        if self.path.exists():
            self.principles = json.loads(self.path.read_text())
        else:
            self.principles = {}

    def _save(self):
        self.path.write_text(json.dumps(self.principles, indent=2))

    def add_principle(self, kernel_category: str, principle: str, evidence: str = ""):
        """Add or update a principle for a kernel category."""
        if kernel_category not in self.principles:
            self.principles[kernel_category] = []

        for existing in self.principles[kernel_category]:
            if existing["principle"] == principle:
                existing["evidence_count"] = existing.get("evidence_count", 1) + 1
                self._save()
                return

        self.principles[kernel_category].append({
            "principle": principle,
            "evidence": evidence,
            "evidence_count": 1,
        })
        self._save()

    def get_principles(self, kernel_category: str) -> list[str]:
        """Get all principles for a kernel category, sorted by evidence count."""
        entries = self.principles.get(kernel_category, [])
        entries.sort(key=lambda x: -x.get("evidence_count", 1))
        return [e["principle"] for e in entries]

    def get_all_principles(self) -> list[str]:
        """Get all principles across all categories."""
        all_p = []
        for cat, entries in self.principles.items():
            for e in entries:
                all_p.append(f"[{cat}] {e['principle']}")
        return all_p

    def format_for_prompt(self, kernel_category: str) -> str:
        """Format principles for injection into agent prompt."""
        direct = self.get_principles(kernel_category)
        if not direct:
            return ""
        lines = ["", "Distilled strategic principles:"]
        for p in direct[:3]:
            lines.append(f"  - {p}")
        return "\n".join(lines)


def seed_principles():
    """Seed the principle store with distilled knowledge from eval runs."""
    store = PrincipleStore()
    store.add_principle(
        "gemm",
        "Triton GEMM cannot beat vendor BLAS (rocBLAS/cuBLAS) for standard shapes. "
        "Focus on quantization-specific fusions or custom operations.",
        "Evidence: 6 runs, max 0.51x vs PyTorch",
    )
    store.add_principle(
        "gemm",
        "For LDS bank conflicts >5/instruction, pad shared memory arrays to reduce conflicts by 60-80%.",
        "Evidence: topk_stage2 LDS padding gave +23%",
    )
    store.add_principle(
        "rope",
        "Direct half-loading (load x1/x2 separately instead of tl.flip/reshape) reduces instruction count. "
        "Provides 2-6% speedup on already-optimized RoPE kernels.",
        "Evidence: rope 2.87x -> 2.94x",
    )
    store.add_principle(
        "attention",
        "OpenEvolve may produce candidates worse than baseline for attention backward kernels. "
        "Manual optimization targeting specific Triton kernels is more effective.",
        "Evidence: nsa_backward OE gave 0.76x, manual gave 1.52x",
    )
    store.add_principle(
        "topk",
        "TopK kernels benefit from manual stage-specific optimization. "
        "Stage1 is latency-bound, Stage2 is LDS-bound -- different strategies needed.",
        "Evidence: topk 1.46x -> 2.07x with targeted optimization",
    )
    store._save()
    return store
