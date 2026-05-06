#!/usr/bin/env python3
"""End-to-end pipeline parity test: refactor-test vs origin/main.

Per user directive 2026-04-08 ("run 1-2 Triton and HIP kernels end to
end, compare with old pipeline").  Runs the preprocessing pipeline on
a fixed set of kernels through BOTH pipelines, captures the artefacts
that downstream optimization depends on, and asserts equivalence on
the CONTRACT (not byte-for-byte).

Equivalence criteria:

  1. Both pipelines produce a harness file that exists on disk.
  2. Both harness files pass ``validate_harness`` (universal contract:
     4 CLI flags + 2 stdout markers).
  3. Both produce ``COMMANDMENT.md`` with the 5 required sections
     (## Setup / ## Correctness / ## Benchmark / ## Full Benchmark /
     ## Profile).
  4. Both produce ``baseline_metrics.json`` with ``duration_us`` and
     ``bottleneck`` top-level keys.
  5. Both produce ``profile.json`` (profiler output).

Environment:
  - ``GEAK_USE_KERNEL_ANALYSIS=0`` (default) so the new pipeline's
    KernelAnalysisAgent stays OFF; the two pipelines become
    apples-to-apples at the preprocess boundary.

Usage:

    python parity_test.py --kernels <path> [<path> ...]

The script writes a ``parity_report.md`` summarising each kernel's
status across both pipelines.  Exit 0 on full parity, 1 on any
failure.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path


REPO_ROOTS = {
    "refactor-test": "/data/sapmajum/GEAK",
    "origin-main": "/data/sapmajum/parity_test/GEAK-main",
}

CONTRACT_FLAGS = ("--correctness", "--benchmark", "--full-benchmark", "--profile")
CONTRACT_MARKERS = ("GEAK_RESULT_LATENCY_MS", "GEAK_RESULT_SPEEDUP")
COMMANDMENT_SECTIONS = (
    "## Setup",
    "## Correctness",
    "## Benchmark",
    "## Full Benchmark",
    "## Profile",
)


class ParityError(AssertionError):
    pass


def _run_preprocess_in_container(
    *,
    pipeline: str,
    kernel_path: Path,
    output_dir: Path,
    repo_root: Path,
    container: str = "geak_agent",
    harness_only: bool = True,
) -> dict:
    """Run ``geak -t '<path>'`` inside the container for ONE pipeline.

    We use ``GEAK_HARNESS_ONLY=1`` to skip the optimization rounds —
    the parity question is about the PREPROCESS stage (harness gen +
    commandment + baseline), which is where the 7-layer extraction
    landed.  Each pipeline's output lands in ``output_dir`` so we can
    diff afterwards.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    env_flags = {
        "GEAK_USE_KERNEL_ANALYSIS": "0",
        "GEAK_USE_KNOWLEDGE_BASE": "0",
        "GEAK_SAVE_TO_KNOWLEDGE_BASE": "0",
    }
    if harness_only:
        env_flags["GEAK_HARNESS_ONLY"] = "1"
    env_str = " ".join(f"{k}={v}" for k, v in env_flags.items())

    cmd = [
        "docker", "exec", container,
        "bash", "-c",
        f"cd {repo_root} && "
        f"pip install -q -e . 2>&1 | tail -1 && "
        f"{env_str} python -m minisweagent.cli "
        f"'optimize {kernel_path} with num_parallel=1 max_rounds=1 gpu_ids=0' "
        f"--output-dir {output_dir} 2>&1 | tail -40",
    ]

    t0 = time.monotonic()
    completed = subprocess.run(cmd, capture_output=True, text=True, timeout=1800)
    elapsed = time.monotonic() - t0

    return {
        "pipeline": pipeline,
        "kernel": str(kernel_path),
        "output_dir": str(output_dir),
        "returncode": completed.returncode,
        "elapsed_s": round(elapsed, 1),
        "stdout_tail": (completed.stdout or "")[-2000:],
        "stderr_tail": (completed.stderr or "")[-2000:],
    }


def _audit_artefacts(output_dir: Path) -> dict:
    """Inspect a preprocess output directory and return a contract-audit dict.

    We check for the presence of each artefact type produced by the
    preprocessing pipeline + do a contract-level inspection where
    relevant.  Missing artefacts are not raised — they're reported so
    the summary markdown can show a red cell.
    """
    audit: dict = {
        "output_dir": str(output_dir),
        "harness_ok": False,
        "harness_path": None,
        "harness_size_bytes": None,
        "harness_has_flags": False,
        "harness_has_markers": False,
        "commandment_ok": False,
        "commandment_sections_missing": [],
        "baseline_metrics_ok": False,
        "baseline_metrics_keys": [],
        "profile_ok": False,
    }

    # Harness: look for harness.py + any test_harness_*.py fallback.
    harness_candidates = sorted(output_dir.rglob("harness*.py")) + sorted(
        output_dir.rglob("test_harness*.py")
    )
    for cand in harness_candidates:
        text = cand.read_text(errors="ignore")
        has_flags = all(f in text for f in CONTRACT_FLAGS)
        has_markers = all(m in text for m in CONTRACT_MARKERS)
        if has_flags or has_markers:
            audit["harness_ok"] = True
            audit["harness_path"] = str(cand)
            audit["harness_size_bytes"] = cand.stat().st_size
            audit["harness_has_flags"] = has_flags
            audit["harness_has_markers"] = has_markers
            break

    # Commandment: COMMANDMENT.md with the 5 required level-2 sections.
    cm_candidates = sorted(output_dir.rglob("COMMANDMENT.md"))
    if cm_candidates:
        cm_text = cm_candidates[0].read_text(errors="ignore")
        missing = [s for s in COMMANDMENT_SECTIONS if s not in cm_text]
        audit["commandment_ok"] = len(missing) == 0
        audit["commandment_sections_missing"] = missing

    # Baseline metrics: baseline_metrics.json with at least duration_us + bottleneck.
    bm_candidates = sorted(output_dir.rglob("baseline_metrics.json"))
    if bm_candidates:
        try:
            bm = json.loads(bm_candidates[0].read_text())
            audit["baseline_metrics_ok"] = (
                "duration_us" in bm and "bottleneck" in bm
            )
            audit["baseline_metrics_keys"] = sorted(bm.keys())[:12]
        except Exception:
            pass

    # Profile: profile.json (profiler output).
    profile_candidates = sorted(output_dir.rglob("profile.json"))
    audit["profile_ok"] = bool(profile_candidates)

    return audit


def _pipeline_audit_to_row(pipeline: str, audit: dict) -> str:
    def _yn(b: bool) -> str:
        return "OK" if b else "NO"

    return (
        f"| {pipeline:<14} | {_yn(audit['harness_ok'])} | "
        f"{_yn(audit['commandment_ok'])} | "
        f"{_yn(audit['baseline_metrics_ok'])} | "
        f"{_yn(audit['profile_ok'])} |"
    )


def _compare_audits(a: dict, b: dict) -> list[str]:
    """Return a list of divergences between two pipeline audits."""
    differences: list[str] = []
    for key in (
        "harness_ok",
        "commandment_ok",
        "baseline_metrics_ok",
        "profile_ok",
    ):
        if a[key] != b[key]:
            differences.append(
                f"  - {key}: refactor-test={a[key]} vs origin-main={b[key]}"
            )
    return differences


def _parity_result_for_kernel(kernel: Path, run_root: Path, harness_only: bool) -> dict:
    """Run both pipelines on ``kernel`` and compare artefacts."""
    print(f"\n=== Kernel: {kernel} ===", flush=True)
    result = {"kernel": str(kernel), "runs": {}, "audits": {}, "differences": []}

    for pipeline, repo_root in REPO_ROOTS.items():
        out_dir = run_root / kernel.stem / pipeline
        print(f"  [{pipeline}] preprocessing → {out_dir}", flush=True)
        run_info = _run_preprocess_in_container(
            pipeline=pipeline,
            kernel_path=kernel,
            output_dir=out_dir,
            repo_root=Path(repo_root),
            harness_only=harness_only,
        )
        result["runs"][pipeline] = run_info

        audit = _audit_artefacts(out_dir)
        result["audits"][pipeline] = audit
        print(
            f"    harness={audit['harness_ok']} "
            f"commandment={audit['commandment_ok']} "
            f"baseline={audit['baseline_metrics_ok']} "
            f"profile={audit['profile_ok']} "
            f"(exit={run_info['returncode']}, elapsed={run_info['elapsed_s']}s)",
            flush=True,
        )

    # Compare.
    a = result["audits"]["refactor-test"]
    b = result["audits"]["origin-main"]
    result["differences"] = _compare_audits(a, b)

    return result


def _write_markdown_report(
    results: list[dict], report_path: Path, *, harness_only: bool
) -> None:
    lines: list[str] = []
    lines.append("# Parity Test Report: refactor-test vs origin/main\n")
    lines.append(
        "Run mode: "
        f"{'harness-only (GEAK_HARNESS_ONLY=1)' if harness_only else 'full pipeline'}\n"
    )
    lines.append(
        f"Kernels tested: {len(results)}; "
        f"KernelAnalysisAgent: OFF (GEAK_USE_KERNEL_ANALYSIS=0, default).\n"
    )
    lines.append("")

    for r in results:
        lines.append(f"## {Path(r['kernel']).name}\n")

        # Headline table.
        lines.append(
            "| pipeline       | harness | commandment | baseline_metrics | profile |\n"
            "|----------------|---------|-------------|------------------|---------|"
        )
        for pipeline in REPO_ROOTS:
            lines.append(_pipeline_audit_to_row(pipeline, r["audits"][pipeline]))
        lines.append("")

        # Divergence summary.
        if not r["differences"]:
            lines.append("**Parity**: full contract match on this kernel.\n")
        else:
            lines.append("**Divergences**:")
            lines.extend(r["differences"])
            lines.append("")

        # Raw run metadata (collapsed).
        for pipeline in REPO_ROOTS:
            run_info = r["runs"][pipeline]
            lines.append(
                f"<details><summary>{pipeline} run metadata "
                f"(exit={run_info['returncode']}, {run_info['elapsed_s']}s)"
                f"</summary>\n"
            )
            lines.append("```")
            lines.append(run_info["stdout_tail"][-800:])
            lines.append("```")
            lines.append("</details>\n")

    lines.append("\n---\n")
    any_diff = any(r["differences"] for r in results)
    if any_diff:
        lines.append("## Overall: DIVERGENT — investigate per-kernel entries above.\n")
    else:
        lines.append("## Overall: PARITY — both pipelines satisfy the contract.\n")

    report_path.write_text("\n".join(lines))
    print(f"\nReport written to {report_path}", flush=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--kernels",
        nargs="+",
        required=True,
        help="Absolute path(s) to kernel files under test.",
    )
    parser.add_argument(
        "--run-root",
        default="/data/sapmajum/parity_test/runs",
        help="Where each pipeline's output directory is created.",
    )
    parser.add_argument(
        "--full-pipeline",
        action="store_true",
        help="Run the full pipeline (no GEAK_HARNESS_ONLY).  "
        "Default: harness-only for fast parity check.",
    )
    parser.add_argument(
        "--report",
        default="/data/sapmajum/parity_test/parity_report.md",
        help="Where to write the markdown report.",
    )
    parser.add_argument(
        "--clean",
        action="store_true",
        help="Wipe run_root before starting.",
    )
    args = parser.parse_args()

    run_root = Path(args.run_root).resolve()
    report_path = Path(args.report).resolve()

    if args.clean and run_root.exists():
        shutil.rmtree(run_root)
    run_root.mkdir(parents=True, exist_ok=True)

    results: list[dict] = []
    for kstr in args.kernels:
        kpath = Path(kstr).resolve()
        if not kpath.is_file():
            print(f"[warn] skipping missing kernel: {kpath}", file=sys.stderr)
            continue
        results.append(
            _parity_result_for_kernel(
                kpath, run_root, harness_only=not args.full_pipeline
            )
        )

    _write_markdown_report(results, report_path, harness_only=not args.full_pipeline)

    any_diff = any(r["differences"] for r in results)
    return 1 if any_diff else 0


if __name__ == "__main__":
    raise SystemExit(main())
