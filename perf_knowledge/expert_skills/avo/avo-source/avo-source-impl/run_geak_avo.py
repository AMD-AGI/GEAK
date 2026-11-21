#!/usr/bin/env python3
"""GEAK-v4-integrated source-level AVO entry point."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import uuid


AVO_DIR = Path(__file__).resolve().parent
# <GEAK>/perf_knowledge/expert_skills/avo/avo-source/avo-source-impl -> <GEAK>
ROOT = AVO_DIR.parents[4] if len(AVO_DIR.parents) > 4 else AVO_DIR
KERNEL_BASE = AVO_DIR / "kernel_base"
# Read from <GEAK>/kernel_workflow/knowledge through one relative symlink each.
LINKED_KNOWLEDGE = (
    "amd_rdna3.md", "amd_rdna4.md", "amd_ryzen.md", "geomean_levers.md",
    "hip_optimization.md", "optimization_strategies.md", "profiling_guide.md",
    "self_monitoring.md", "triton_optimization.md", "wrapper_optimization.md",
)
RUNS_ROOT = ROOT.parent / f"{ROOT.name}-runs"
LOW_LEVEL = {"asm", "amdgcn", "gcn", "ir", "llvm-ir", "llvm_ir", "ll", "ttgir"}


def load(path: Path) -> dict:
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return value


def node_binary() -> str:
    for candidate in (
        os.environ.get("GEAK_NODE_BIN"),
        shutil.which("node"),
    ):
        if not candidate or not Path(candidate).is_file():
            continue
        try:
            major = int(subprocess.check_output(
                [candidate, "-p", "process.versions.node.split('.')[0]"],
                text=True,
                timeout=5,
            ).strip())
        except (OSError, ValueError, subprocess.SubprocessError):
            continue
        if major >= 18:
            return candidate
    raise RuntimeError("Node.js >=18 not found; set GEAK_NODE_BIN")


def check_geak_layout() -> None:
    """AVO reads these from the GEAK checkout it lives in; fail before any agent starts."""
    required = [
        ROOT / "perf_knowledge",
        ROOT / "kernel_workflow" / "scripts" / "gpu_lock.sh",
        KERNEL_BASE / "scripts" / "gpu_lock.sh",
        KERNEL_BASE / "scripts" / "profile_kernel.sh",
        KERNEL_BASE / "roles" / "benchmark_engineer.md",
        KERNEL_BASE / "roles" / "profile_engineer.md",
        KERNEL_BASE / "roles" / "tech_lead.md",
        KERNEL_BASE / "knowledge" / "amd_instinct.md",
        *(KERNEL_BASE / "knowledge" / name for name in LINKED_KNOWLEDGE),
    ]
    missing = [str(path) for path in required if not path.exists()]
    not_links = [
        name for name in LINKED_KNOWLEDGE
        if (KERNEL_BASE / "knowledge" / name).exists()
        and not (KERNEL_BASE / "knowledge" / name).is_symlink()
    ]
    if not_links:
        raise ValueError(
            "this checkout did not keep symbolic links (kernel_base/knowledge/"
            + ", ".join(not_links) + "); clone GEAK with symlink support"
        )
    if missing:
        raise ValueError(
            "AVO must run from inside a complete GEAK checkout; missing: "
            + ", ".join(missing)
        )


def make_config(handoff: dict) -> dict:
    kernel = Path(str(handoff.get("kernel_path") or "")).expanduser()
    if not kernel.is_absolute() or not kernel.exists():
        raise ValueError("kernel_path must be an existing absolute path")
    check_geak_layout()
    nested_args = handoff.get("args") or {}
    if not isinstance(nested_args, dict):
        raise ValueError("args must be a JSON object when provided")
    level = str(
        handoff.get("avo_language_level")
        or handoff.get("language_level")
        or nested_args.get("avo_language_level")
        or nested_args.get("language_level")
        or "source"
    ).lower()
    target = str(
        handoff.get("target_language") or nested_args.get("target_language") or ""
    ).lower()
    low_level_flag = any(
        str(handoff.get(key) or nested_args.get(key) or "").lower() == "true"
        for key in ("asm_mode", "ir_mode", "ir_feedback")
    )
    if level != "source" or target in LOW_LEVEL or low_level_flag:
        raise ValueError("integrated AVO supports source-level optimization only")

    exp_root = Path(str(
        handoff.get("exp_root") or RUNS_ROOT / "avo-source"
    )).expanduser().resolve()
    args = dict(nested_args)
    for key in (
        "budget_s", "final_phase_reserve_s", "gpu_ids", "task", "operator",
        "operator_directions",
        "early_stop", "min_commits", "max_steps", "verify_repeats",
        "apply_to_original", "use_profiling", "profiling_after_step",
        "start_policy", "start_k", "escalate_workers", "target_language",
        "kernel_env_script", "eval_dir", "hw", "final_reproduction_tolerance",
    ):
        if key in handoff:
            args[key] = handoff[key]
    gpu_ids = args.get("gpu_ids", "0")
    if isinstance(gpu_ids, list):
        gpu_ids = ",".join(str(gpu) for gpu in gpu_ids)
    gpu_ids = str(gpu_ids)
    gpu_count = len([gpu for gpu in gpu_ids.split(",") if gpu.strip()])
    args["gpu_ids"] = gpu_ids
    args.setdefault("operator", "team")
    args.setdefault("operator_directions", max(2, gpu_count))
    args.setdefault("start_policy", "diverse")
    args.setdefault("start_k", args["operator_directions"])
    args.update({
        "kernel_path": str(kernel),
        "workflow_dir": str(AVO_DIR),
        "kernel_workflow_dir": str(KERNEL_BASE),
        "kernel_knowledge_dir": str(ROOT / "perf_knowledge"),
        "exp_root": str(exp_root),
        "avo_language_level": "source",
        "asm_mode": "false",
        "ir_mode": "false",
        "ir_feedback": "false",
    })
    return {
        "scriptPath": str(AVO_DIR / "avo_workflow.js"),
        "claude_bin": str(
            handoff.get("claude_bin")
            or os.environ.get("GEAK_CLAUDE_BIN")
            or shutil.which("claude")
            or "claude"
        ),
        "claude_home": str(
            handoff.get("claude_home")
            or os.environ.get("GEAK_CLAUDE_HOME")
            or Path.home()
        ),
        "root_dir": str(Path(str(
            handoff.get("root_dir")
            or os.path.commonpath([str(ROOT), str(kernel.resolve())])
        )).resolve()),
        "model": str(handoff.get("model") or os.environ.get("GEAK_CLAUDE_MODEL") or ""),
        "agent_timeout_ms": int(handoff.get("agent_timeout_ms") or 2_400_000),
        "agent_retries": int(handoff.get("agent_retries") or 1),
        "args": args,
    }


def result_for(config: dict, code: int, error: str = "") -> dict:
    raw_path = Path(config["args"]["exp_root"]) / "_shim_logs" / "result.json"
    try:
        raw = load(raw_path) if raw_path.is_file() else {}
    except (OSError, ValueError, json.JSONDecodeError):
        raw = {}
    expected_run_id = str(config.get("run_id") or "")
    if expected_run_id and raw.get("_shim_run_id") != expected_run_id:
        raw = {}
    # A failed shim invocation must never forward a previous campaign's contract
    # from the same exp_root. The raw file is diagnostic only on non-zero exit.
    if code:
        raw = {}
    validation = str(raw.get("validation_status") or "")
    if code or not raw:
        status = "error"
    elif validation == "accepted":
        status = "ok"
    elif validation in {"no_gain", "no_significant_gain"}:
        status = "no_gain"
    elif validation == "reproduction_mismatch":
        status = "reproduction_mismatch"
    else:
        status = "error"
    return {
        "schema_version": 1,
        "status": status,
        "engine": "avo-source",
        "kernel_path": config["args"]["kernel_path"],
        "eval_dir": raw.get("eval_dir", ""),
        "validation_status": validation or "error",
        "final_speedup": raw.get("final_geomean", 0),
        "final_patch": raw.get("final_patch", ""),
        "output_bundle": raw.get("output_bundle", ""),
        "output_artifacts": raw.get("output_artifacts", []),
        "best_version": raw.get("best_version", ""),
        "num_commits": raw.get("num_commits", 0),
        "steps": raw.get("steps", 0),
        "report_path": raw.get("report_path", ""),
        "formal_report_path": raw.get("formal_report_path", ""),
        "roofline_fraction": raw.get("roofline_fraction"),
        "roofline_regime": raw.get("roofline_regime", "unknown"),
        "roofline_basis": raw.get("roofline_basis", "unavailable"),
        "roofline_evidence_path": raw.get("roofline_evidence_path", ""),
        "practical_roofline_fraction": raw.get("practical_roofline_fraction"),
        "practical_roofline_regime": raw.get(
            "practical_roofline_regime", "unknown"),
        "practical_roofline_evidence_path": raw.get(
            "practical_roofline_evidence_path", ""),
        "raw_result_path": str(raw_path),
        **({"error": error or (
            f"run_avo.mjs exited {code}" if code
            else f"final validation status: {validation or 'missing'}"
        )} if status == "error" else {}),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run source-level AVO on GEAK-v4")
    parser.add_argument("handoff", type=Path)
    parser.add_argument("result", type=Path)
    parser.add_argument("--dry-run", action="store_true")
    ns = parser.parse_args(argv)
    try:
        config = make_config(load(ns.handoff))
        if ns.dry_run:
            print(json.dumps(config, indent=2))
            return 0
        exp = Path(config["args"]["exp_root"])
        exp.mkdir(parents=True, exist_ok=True)
        config["run_id"] = uuid.uuid4().hex
        raw_result = exp / "_shim_logs" / "result.json"
        try:
            raw_result.unlink(missing_ok=True)
        except OSError:
            # The run-id check below still makes an unremovable old contract inert.
            pass
        cfg = exp / "avo_run_config.json"
        cfg.write_text(json.dumps(config, indent=2) + "\n")
        completed = subprocess.run(
            [node_binary(), str(AVO_DIR / "run_avo.mjs"), str(cfg)],
            check=False,
        )
        result = result_for(config, completed.returncode)
        ns.result.parent.mkdir(parents=True, exist_ok=True)
        ns.result.write_text(json.dumps(result, indent=2) + "\n")
        return 0 if result["status"] in {"ok", "no_gain"} else 1
    except Exception as exc:  # noqa: BLE001
        payload = {
            "schema_version": 1, "status": "error",
            "engine": "avo-source", "error": str(exc),
        }
        try:
            ns.result.parent.mkdir(parents=True, exist_ok=True)
            ns.result.write_text(json.dumps(payload, indent=2) + "\n")
        except OSError:
            pass
        print(f"run_geak_avo.py: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
