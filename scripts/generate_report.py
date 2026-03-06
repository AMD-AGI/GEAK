#!/usr/bin/env python3
"""Generate detailed experiment report from results.

Usage: python3 generate_report.py <experiment_dir> <output_report_path>

Produces a markdown report with:
  1. Summary table (per-kernel speedups, patch sizes, wall time)
  2. Measurement verification (artifact detection)
  3. Memory interaction log (if applicable)
  4. Qualitative samples
  5. Findings
  6. Raw data appendix
"""

from __future__ import annotations

import json
import math
import os
import re
import sys
from glob import glob
from pathlib import Path


def load_kernel_results(exp_dir: Path) -> list[dict]:
    results = []
    results_json = exp_dir / "results_json"
    if results_json.is_dir():
        for f in sorted(results_json.glob("*.json")):
            with open(f) as fh:
                results.append(json.load(fh))
    return results


def geo_mean(values: list[float]) -> float:
    if not values:
        return 0.0
    return math.exp(sum(math.log(max(v, 0.001)) for v in values) / len(values))


def detect_artifacts(exp_dir: Path, kernel: str) -> list[str]:
    """Check for measurement artifacts in a kernel's results."""
    artifacts = []
    kernel_dir = exp_dir / kernel
    if not kernel_dir.is_dir():
        return ["No results directory found"]

    for br_file in kernel_dir.rglob("best_results.json"):
        try:
            br = json.loads(br_file.read_text())
            speedup = float(br.get("best_patch_speedup", 0))
            patch_file = br.get("best_patch_file", "")
            patch_size = br.get("best_patch_size_bytes", -1)

            if patch_size == -1 and patch_file and Path(patch_file).is_file():
                patch_size = Path(patch_file).stat().st_size

            if speedup > 1.0 and patch_size == 0:
                artifacts.append(f"ARTIFACT: {speedup:.2f}x with 0-byte patch in {br_file.parent.name}")
            elif speedup > 5.0:
                artifacts.append(f"SUSPECT: {speedup:.2f}x unusually high in {br_file.parent.name}")
        except Exception:
            pass
    return artifacts


def extract_patch_summary(exp_dir: Path, kernel: str) -> str:
    """Get a summary of the best patch's code changes."""
    kernel_dir = exp_dir / kernel
    best_speedup = 0.0
    best_patch_path = None

    for br_file in kernel_dir.rglob("best_results.json"):
        try:
            br = json.loads(br_file.read_text())
            sp = float(br.get("best_patch_speedup", 0))
            pf = br.get("best_patch_file", "")
            if sp > best_speedup and pf and Path(pf).is_file() and Path(pf).stat().st_size > 0:
                best_speedup = sp
                best_patch_path = pf
        except Exception:
            pass

    if not best_patch_path or not Path(best_patch_path).is_file():
        return "No valid patch found"

    try:
        content = Path(best_patch_path).read_text()
        lines = content.splitlines()
        if len(lines) > 30:
            return "\n".join(lines[:30]) + f"\n... ({len(lines) - 30} more lines)"
        return content
    except Exception:
        return "Could not read patch"


def generate_report(exp_dir: Path, output_path: Path, exp_id: str):
    results = load_kernel_results(exp_dir)
    summary_path = exp_dir / "experiment_summary.json"
    summary = {}
    if summary_path.exists():
        summary = json.loads(summary_path.read_text())

    config = summary.get("config", "")
    speedups = [r["speedup"] for r in results if r.get("speedup", 0) > 0]
    gmean = geo_mean(speedups) if speedups else 0.0

    lines = [
        f"# Experiment Report: {exp_id}",
        "",
        f"**Config**: `{config}`",
        f"**Kernels**: {len(results)}",
        f"**Geo-mean speedup**: {gmean:.4f}x",
        "",
        "## Section 1: Summary Table",
        "",
        "| Kernel | Speedup | Patch Size | Rounds | Wall Time | Non-Empty Patches |",
        "|--------|---------|------------|--------|-----------|-------------------|",
    ]

    for r in results:
        kernel = r.get("kernel", "?")
        sp = r.get("speedup", 0)
        psz = r.get("patch_size_bytes", 0)
        rounds = r.get("rounds", 0)
        wt = r.get("wall_time_sec", 0)
        ne = r.get("non_empty_patches", 0)
        wt_str = f"{wt // 60}m {wt % 60}s" if wt > 0 else "N/A"
        psz_str = f"{psz / 1024:.1f} KB" if psz > 1024 else f"{psz} B"
        lines.append(f"| {kernel} | {sp:.4f}x | {psz_str} | {rounds} | {wt_str} | {ne} |")

    lines.append(f"| **GEO-MEAN** | **{gmean:.4f}x** | | | | |")

    lines.extend([
        "",
        "## Section 2: Measurement Verification",
        "",
    ])

    for r in results:
        kernel = r.get("kernel", "?")
        sp = r.get("speedup", 0)
        psz = r.get("patch_size_bytes", 0)
        artifacts = detect_artifacts(exp_dir, kernel)

        if sp <= 1.0:
            verdict = "NO IMPROVEMENT"
        elif psz == 0:
            verdict = "MEASUREMENT ARTIFACT (empty patch)"
        elif sp > 5.0 and not artifacts:
            verdict = "SUSPECT -- needs manual review"
        elif artifacts:
            verdict = f"FLAGGED: {'; '.join(artifacts)}"
        else:
            verdict = "REAL IMPROVEMENT (pending patch review)"

        lines.append(f"### {kernel}: {sp:.4f}x -- {verdict}")
        if sp > 1.0:
            patch_summary = extract_patch_summary(exp_dir, kernel)
            lines.extend([
                "",
                "```diff",
                patch_summary[:2000],
                "```",
                "",
            ])

    lines.extend([
        "",
        "## Section 3: Memory Interaction Log",
        "",
    ])

    if "DISABLE" in config:
        lines.append("*Memory disabled for this experiment.*")
    else:
        lines.append("Memory interaction details from agent logs:")
        lines.append("")
        for r in results:
            kernel = r.get("kernel", "?")
            log_file = exp_dir / "logs" / f"{kernel}.log"
            if log_file.is_file():
                try:
                    log_text = log_file.read_text()
                    mem_refs = len(re.findall(r"(?i)memory|past outcome|prior run", log_text))
                    lines.append(f"- **{kernel}**: {mem_refs} memory references in agent log")
                except Exception:
                    lines.append(f"- **{kernel}**: Log unreadable")
            else:
                lines.append(f"- **{kernel}**: No log file found")

    lines.extend([
        "",
        "## Section 4: Qualitative Samples",
        "",
        "*To be filled after manual review of top-3 kernels with highest speedup.*",
        "",
        "## Section 5: Findings",
        "",
    ])

    real_improvements = [r for r in results if r.get("speedup", 0) > 1.0 and r.get("patch_size_bytes", 0) > 0]
    artifacts_found = [r for r in results if r.get("speedup", 0) > 1.0 and r.get("patch_size_bytes", 0) == 0]
    no_improvement = [r for r in results if r.get("speedup", 0) <= 1.0]

    lines.extend([
        f"- **Real improvements**: {len(real_improvements)} / {len(results)} kernels",
        f"- **Measurement artifacts**: {len(artifacts_found)} / {len(results)} kernels",
        f"- **No improvement**: {len(no_improvement)} / {len(results)} kernels",
        f"- **Geo-mean (real only)**: {geo_mean([r['speedup'] for r in real_improvements]):.4f}x" if real_improvements else "",
        "",
        "## Section 6: Raw Data Appendix",
        "",
        f"- Experiment directory: `{exp_dir}`",
        f"- Log directory: `{exp_dir / 'logs'}`",
        f"- Results JSON: `{exp_dir / 'results_json'}`",
        "",
    ])

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines))
    print(f"Report written to: {output_path}")


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python3 generate_report.py <experiment_dir> <output_report_path> [exp_id]")
        sys.exit(1)
    exp_dir = Path(sys.argv[1])
    output_path = Path(sys.argv[2])
    exp_id = sys.argv[3] if len(sys.argv) > 3 else exp_dir.name
    generate_report(exp_dir, output_path, exp_id)
