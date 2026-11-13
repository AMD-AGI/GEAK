"""KernelFusion switch and knobs: the only KernelFusion code interface/run_e2e.py calls.

KernelFusion is OFF by default. Switched on, e2e_workflow.js adds a fusion profile beside Profile's Top-N,
Strategize routes its regions into the kernel track, and the Milestone loop builds, optimizes and A/B-tests
each region like any kernel candidate (e2e_workflow/knowledge/kernel_fusion_integration.md). This module only
turns the switch and its knobs into workflow args. Switch it on with:

    handoff  "kernel_fusion": true | {...settings}  (a settings object switches it on unless "enabled": false)
    env      GEAK_KERNEL_FUSION=1|0            (overrides the handoff)
    CLI      run_e2e.py ... --kernel-fusion | --no-kernel-fusion   (overrides both)

Off, workflow_args() is {}: the workflow gets exactly the args it gets without KernelFusion. A knob that
cannot be used makes the workflow record a KernelFusion error and run no fusion profile; it never stops the
run.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

GEAK_ROOT = Path(__file__).resolve().parent.parent
# The fusion-analysis skill every analysis follows (fusion levels, requirements, procedure, estimation, report
# contents). Run-specific prompts (deployment, workload, extra constraints) are read alongside it.
FUSION_STRATEGY = GEAK_ROOT / "e2e_workflow" / "knowledge" / "kernel_fusion_skills" / "fusion-analysis" / "SKILL.md"

# handoff key -> (env var, e2e_workflow.js arg)
_KNOBS = {
    "strategy": ("GEAK_KF_STRATEGY", None),
    "prompt": ("GEAK_KF_PROMPT", "kf_prompt"),
    "trace_dir": ("GEAK_KF_TRACE_DIR", "kf_trace_dir"),
    "workload_summary": ("GEAK_KF_WORKLOAD_SUMMARY", "kf_workload_summary"),
    "topk_json": ("GEAK_KF_TOPK_JSON", "kf_topk_json"),
    "source_dirs": ("GEAK_KF_SOURCE_DIRS", "kf_source_dirs"),
    "implementable_levels": ("GEAK_KF_IMPLEMENTABLE_LEVELS", "kf_implementable_levels"),
    "unsupported_patterns": ("GEAK_KF_UNSUPPORTED_PATTERNS", "kf_unsupported_patterns"),
    "task": ("GEAK_KF_TASK", "kf_task"),
}
_PATH_ARGS = ("kf_prompt", "kf_trace_dir", "kf_workload_summary", "kf_topk_json")


def _truthy(v: Any) -> bool:
    return str(v).strip().lower() in ("1", "true", "yes", "on")


def config(h: dict, flags: set[str] | frozenset = frozenset()) -> dict | None:
    """The KernelFusion config when the switch is on, else None (the default). CLI > env > handoff. A handoff
    ``true`` or settings object switches it on (an object stays off only with "enabled": false)."""
    raw = h.get("kernel_fusion")
    cfg: dict[str, Any] = dict(raw) if isinstance(raw, dict) else {}
    cfg["enabled"] = _truthy(cfg.get("enabled", True)) if isinstance(raw, dict) else (raw is not None and _truthy(raw))
    if os.environ.get("GEAK_KERNEL_FUSION", "").strip():
        cfg["enabled"] = _truthy(os.environ["GEAK_KERNEL_FUSION"])
    if "--kernel-fusion" in flags:
        cfg["enabled"] = True
    if "--no-kernel-fusion" in flags:
        cfg["enabled"] = False
    for key, (env, _arg) in _KNOBS.items():
        if os.environ.get(env, "").strip():
            cfg[key] = os.environ[env].strip()
    return cfg if cfg["enabled"] else None


def prompt_files(cfg: dict) -> list[str]:
    """The fusion strategy first (``strategy``: another file, or off with false/""), then the run's own
    prompts (``prompt``, comma-separated), each file once."""
    strategy = cfg.get("strategy", True)
    files = [] if strategy in (False, "", "0", "false", "off") else [
        str(FUSION_STRATEGY if strategy in (True, None, "1", "true", "on") else strategy)]
    files += [p for p in str(cfg.get("prompt") or "").split(",") if p.strip()]
    out, seen = [], set()
    for f in files:
        r = str(Path(f.strip()).expanduser().resolve())
        if r not in seen:
            seen.add(r)
            out.append(r)
    return out


# A given topk.json travels in the args: pass only what the workflow reads.
CAND_KEYS = ("id", "rank", "level", "region", "kernels", "phases", "occurrences", "m_range", "seam", "inputs_outputs",
             "launches_before", "launches_after", "region_pct_device_time", "est_gain_pct_expected", "difficulty",
             "bitwise_exact", "exclusive_with", "language", "fallback", "fused_design")


def slim_topk(d: dict) -> dict:
    """Every candidate (the System Architect judges them all), each with only CAND_KEYS."""
    cands = {str(c.get("id")): c for c in d.get("candidates") or [] if isinstance(c, dict)}
    return {"candidates": [{k: c[k] for k in CAND_KEYS if k in c} for c in cands.values()],
            "recommended": [str(i) for i in d.get("recommended") or []]}


def _read_json(p: Path) -> dict | None:
    try:
        d = json.loads(p.read_text())
        return d if isinstance(d, dict) else None
    except (OSError, ValueError):
        return None


def validate(args: dict) -> list[str]:
    """Problems that make the fusion profile pointless to run (the analysis has nothing to work from)."""
    probs = []
    for k in _PATH_ARGS:
        for p in str(args.get(k) or "").split(","):
            if p and not Path(p).exists():
                probs.append(f"{k} does not exist: {p}")
    if args.get("kf_topk_json") and Path(args["kf_topk_json"]).exists() and not (args.get("kf_topk") or {}).get("candidates"):
        probs.append(f"kf_topk_json has no candidates: {args['kf_topk_json']}")
    return probs


def build_args(cfg: dict) -> dict:
    a: dict[str, Any] = {"kernel_fusion": "true"}
    for key, (_env, arg) in _KNOBS.items():
        if arg and cfg.get(key) not in (None, ""):
            a[arg] = cfg[key]
    a["kf_prompt"] = ",".join(prompt_files(cfg))
    for path_key in _PATH_ARGS:
        if a.get(path_key):
            # kf_prompt may name several files (comma-separated): all of them drive the analysis.
            a[path_key] = ",".join(str(Path(p.strip()).expanduser().resolve())
                                   for p in str(a[path_key]).split(",") if p.strip())
    if a.get("kf_topk_json"):
        # a finished analysis is passed in whole: the workflow script cannot read files
        d = _read_json(Path(a["kf_topk_json"])) or {}
        a["kf_topk"] = slim_topk(d)
    return a


def workflow_args(h: dict, flags: set[str] | frozenset = frozenset()) -> dict:
    """The args run_e2e adds to the e2e workflow's: {} when the switch is off (the default). Switched on,
    a knob that cannot be used is handed over as kf_config_errors: the workflow records it and runs no fusion
    profile."""
    cfg = config(h, flags)
    if not cfg:
        return {}
    try:
        a = build_args(cfg)
    except Exception as e:  # a bad knob is recorded, never ends the run
        return {"kernel_fusion": "true", "kf_config_errors": f"{type(e).__name__}: {e}"[:500]}
    probs = validate(a)
    if probs:
        a["kf_config_errors"] = "; ".join(probs)[:2000]
    return a
