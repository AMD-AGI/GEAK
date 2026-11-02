"""KernelFusion stage driver: the only KernelFusion code interface/run_e2e.py calls.

The stage is ON by default and runs to completion BEFORE the main e2e workflow
(baseline -> trace -> topk -> author -> apply back -> the e2e workflow). Switch it off with:

    handoff  "kernel_fusion": false | {"enabled": false}   (an object configures it: {"prompt": "...", ...})
    env      GEAK_KERNEL_FUSION=0|1            (overrides the handoff)
    CLI      run_e2e.py ... --kernel-fusion | --no-kernel-fusion   (overrides both)

It runs kernel_fusion/kf_workflow.js through run_e2e's own invoke_workflow (Claude SDK / CLI / standalone
runtime, same env, same bench client), on the same mapped args the main workflow gets. When the stage's
A/B accepted a fused overlay, that overlay becomes the main workflow's initial overlay, so the main
workflow measures its own baseline on the fused stack. Nothing else about the main workflow changes, and
a failed or empty stage leaves its args exactly as they were.
"""
from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any

KF_HOME = Path(__file__).resolve().parent
KF_SCRIPT = KF_HOME / "kf_workflow.js"
DEFAULT_BUDGET_S = 12 * 3600
SCHEMA = "geak.kernel_fusion/1"

# handoff key / env var -> kf_workflow.js arg
# The fusion strategy every analysis follows (fusion levels, selection principles, analysis steps, output
# contract). Run-specific prompts (deployment, workload, extra constraints) are read alongside it.
FUSION_STRATEGY = KF_HOME / "fusion_strategy.md"

_KNOBS = {
    "strategy": ("GEAK_KF_STRATEGY", None),
    "prompt": ("GEAK_KF_PROMPT", "kf_prompt"),
    "trace_dir": ("GEAK_KF_TRACE_DIR", "kf_trace_dir"),
    "workload_summary": ("GEAK_KF_WORKLOAD_SUMMARY", "kf_workload_summary"),
    "baseline_dir": ("GEAK_KF_BASELINE_DIR", "kf_baseline_dir"),
    "topk_json": ("GEAK_KF_TOPK_JSON", "kf_topk_json"),
    "reuse_journal": ("GEAK_KF_REUSE_JOURNAL", None),
    "source_dirs": ("GEAK_KF_SOURCE_DIRS", "kf_source_dirs"),
    "steps": ("GEAK_KF_STEPS", "kf_steps"),
    "author_k": ("GEAK_KF_AUTHOR_K", "kf_author_k"),
    "validation_mode": ("GEAK_KF_VALIDATION_MODE", "kf_validation_mode"),
    "validation_replicas": ("GEAK_KF_VALIDATION_REPLICAS", "kf_validation_replicas"),
    "task": ("GEAK_KF_TASK", "kf_task"),
    "budget_s": ("GEAK_KF_BUDGET_S", None),
    "kf_dir": ("GEAK_KF_DIR", "kf_dir"),
}
# Main-workflow-only args that mean nothing (or something wrong) to the stage.
_MAIN_ONLY = ("eval_dir", "phases", "state", "tracelens", "time_budget_s", "final_reserve_s",
              "config_tune", "tuning_skillset", "tuning_kb", "apply_to_original")


def _truthy(v: Any) -> bool:
    return str(v).strip().lower() in ("1", "true", "yes", "on")


def config(h: dict, flags: set[str] | frozenset = frozenset()) -> dict | None:
    """The stage config when the switch is on (the default), else None. CLI > env > handoff."""
    raw = h.get("kernel_fusion")
    cfg: dict[str, Any] = dict(raw) if isinstance(raw, dict) else {}
    cfg["enabled"] = _truthy(cfg.get("enabled", True)) if isinstance(raw, dict) else (raw is None or _truthy(raw))
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


def stage_budget_s(cfg: dict, timeout_s: int) -> int:
    if cfg.get("budget_s"):
        return int(cfg["budget_s"])
    return int(min(DEFAULT_BUDGET_S, max(timeout_s // 3, 1)))


def kf_dir_for(cfg: dict, ps_args: dict) -> Path:
    if cfg.get("kf_dir"):
        return Path(cfg["kf_dir"])
    main = Path(ps_args["eval_dir"])            # <exp_root>/e2e_<model>_<ts>_<id>Z
    return main.with_name("kf_" + main.name.removeprefix("e2e_"))


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


def build_args(cfg: dict, ps_args: dict, budget_s: int) -> dict:
    a = {k: v for k, v in ps_args.items() if k not in _MAIN_ONLY}
    for key, (_env, arg) in _KNOBS.items():
        if arg and cfg.get(key) not in (None, ""):
            a[arg] = cfg[key]
    a["kf_prompt"] = ",".join(prompt_files(cfg))
    for path_key in ("kf_prompt", "kf_trace_dir", "kf_workload_summary", "kf_baseline_dir", "kf_topk_json"):
        if a.get(path_key):
            # kf_prompt may name several files (comma-separated): all of them drive the analysis.
            a[path_key] = ",".join(str(Path(p.strip()).expanduser().resolve())
                                   for p in str(a[path_key]).split(",") if p.strip())
    if a.get("kf_topk_json"):
        # a finished analysis is passed in whole: the workflow script cannot read files, and an agent
        # asked to echo a large topk.json back may drop its candidates
        d = _read_json(Path(a["kf_topk_json"])) or {}
        a["kf_topk"] = slim_topk(d, int(a.get("kf_author_k") or 3))
    if cfg.get("reuse_journal"):
        reuse: dict = {}
        for j in str(cfg["reuse_journal"]).split(","):          # several runs: a later journal wins per label
            if j.strip():
                reuse.update(reused_results(j.strip()))
        a["kf_reuse"] = {k: slim_result(v) for k, v in reuse.items()}
    a["kf_dir"] = str(kf_dir_for(cfg, ps_args))
    a["time_budget_s"] = int(budget_s)
    return a


# The driver agent re-types the args into its Workflow call, and a large payload comes out truncated: pass
# only what kf_workflow.js reads.
CAND_KEYS = ("id", "rank", "level", "region", "kernels", "phases", "occurrences", "m_range", "seam", "inputs_outputs",
             "launches_before", "launches_after", "region_pct_device_time", "est_gain_pct_expected", "difficulty",
             "bitwise_exact", "exclusive_with", "language", "fallback", "fused_design")
UT_KEYS = ("task_dir", "editable", "baseline_frozen", "unittest_smoke", "smoke", "target_callable", "short_name")
AUTHOR_KEYS = ("final_patch", "work_dir", "overlay_dir", "banner", "isolated_speedup", "correctness", "launches_after")


def slim_topk(d: dict, author_k: int) -> dict:
    """The recommended candidates the stage implements, plus every candidate they name (fallback / exclusive)."""
    cands = {str(c.get("id")): c for c in d.get("candidates") or [] if isinstance(c, dict)}
    rec = [str(i) for i in d.get("recommended") or []][:max(author_k, 1)]
    keep = set(rec)
    for i in rec:
        c = cands.get(i) or {}
        keep.update(str(x) for x in [c.get("fallback"), *(c.get("exclusive_with") or [])] if x is not None)
    return {"candidates": [{k: c[k] for k in CAND_KEYS if k in c} for i, c in cands.items() if i in keep],
            "recommended": rec}


def slim_result(r: dict) -> dict:
    """A reused prepare / author result with only the fields the stage reads."""
    if "members" in r:                                           # the correctness screen
        return {"members": [{k: m.get(k) for k in ("short_name", "correctness", "engaged")} | {"reason": str(m.get("reason") or "")[:200]}
                            for m in r.get("members") or [] if isinstance(m, dict)]}
    if "prepared" in r:
        p = r.get("prepared") or {}
        cands = [{"short_name": k.get("short_name"),
                  "prepared_ut": {"phase": (k.get("prepared_ut") or {}).get("phase"),
                                  "result": {x: ((k.get("prepared_ut") or {}).get("result") or {}).get(x) for x in UT_KEYS}}}
                 for k in [*(p.get("head_candidates") or []), *(p.get("kernel_candidates") or [])] if isinstance(k, dict)]
        why = [{"reason": str(f.get("reason") or f.get("notes") or f.get("status") or "")[:300]}
               for f in [*(p.get("failed_tasks") or []), *(p.get("deferred_tasks") or [])] if isinstance(f, dict)]
        return {"delivery_path": r.get("delivery_path", ""),
                "prepared": {"head_candidates": [], "kernel_candidates": cands, "failed_tasks": why}}
    out = {k: r[k] for k in AUTHOR_KEYS if k in r}
    out["notes"] = str(r.get("notes") or "")[:300]
    return out


REUSABLE = ("kf:kernel-extract prepare ", "kf:author ", "kf:integrate screen")


def reused_results(journal: str) -> dict:
    """Finished UT-prepare / author agent results of an earlier run of this stage, by label, read from that run's
    Workflow journal.jsonl ("started" rows carry the label, "result" rows the agent's return)."""
    labels, out, old_dir = {}, {}, ""
    try:
        lines = Path(journal).read_text(encoding="utf-8").splitlines()
    except OSError:
        return {}
    for line in lines:
        try:
            row = json.loads(line)
        except ValueError:
            continue
        if row.get("type") == "started":
            labels[row.get("key")] = row.get("label") or ""
        elif row.get("type") == "result" and isinstance(row.get("result"), dict):
            label = labels.get(row.get("key"), "")
            if label == "kf:director:setup":
                old_dir = str(row["result"].get("eval_dir") or "")
            if label.startswith(REUSABLE):
                out[label] = row["result"]
    # that run's eval dir may have been archived beside its journal (<journal dir>/exp/<name>): repoint paths
    moved = Path(journal).parent / "exp" / Path(old_dir).name if old_dir else None
    if out and old_dir and not Path(old_dir).exists() and moved and moved.exists():
        out = json.loads(json.dumps(out).replace(old_dir, str(moved)))
    return out


def validate(args: dict) -> list[str]:
    """Problems that make the stage pointless to start (the analysis has nothing to work from)."""
    probs = []
    for k in ("kf_prompt", "kf_trace_dir", "kf_workload_summary", "kf_baseline_dir", "kf_topk_json"):
        for p in str(args.get(k) or "").split(","):
            if p and not Path(p).exists():
                probs.append(f"{k} does not exist: {p}")
    if args.get("kf_topk_json") and Path(args["kf_topk_json"]).exists() and not (args.get("kf_topk") or {}).get("candidates"):
        probs.append(f"kf_topk_json has no candidates: {args['kf_topk_json']}")
    return probs


def build_prompt(G: dict, args: dict) -> str:
    """The driver prompt: run_e2e.build_prompt's contract, pointed at kf_workflow.js."""
    ret = f"{args['kf_dir']}/workflow_return.json"
    return (
        G["PROCESS_SAFETY"]
        + "Invoke the Workflow tool exactly once with:\n"
        f'  scriptPath: "{KF_SCRIPT}"\n'
        f"  args: {json.dumps(args)}\n"
        "CRITICAL: pass `args` as a real JSON OBJECT (a mapping), NOT as a JSON-encoded string, and copy it EXACTLY: "
        "every key and every element, nothing shortened or omitted (kf_topk / kf_reuse are data the stage consumes).\n"
        "This is the GEAK KernelFusion stage (Setup -> Trace -> TopK -> Author -> ApplyBack). When the tool "
        f'returns, write its full return value to "{ret}" with the Write tool if that file does not exist, '
        "then print EXACTLY ONE final line of compact JSON that is the Workflow tool's full return value. "
        "Print nothing after the JSON line. If the Workflow tool fails, do NOT edit any file, fix the script or "
        "launch it again: print the error and stop (the caller records it and continues)."
    )


def _invoke(G: dict, args: dict, budget_s: int) -> dict:
    saved = G["E2E_SCRIPT"]
    G["E2E_SCRIPT"] = KF_SCRIPT     # the standalone-runtime path launches run_e2e's global E2E_SCRIPT
    try:
        # The JS stops starting work at 85% of time_budget_s; the slack lets in-flight A/B legs land.
        return G["invoke_workflow"](build_prompt(G, args), int(budget_s * 1.1) + 900, args["kf_dir"], ps_args=args)
    finally:
        G["E2E_SCRIPT"] = saved


def _read_json(p: Path) -> dict | None:
    try:
        d = json.loads(p.read_text())
        return d if isinstance(d, dict) else None
    except (OSError, ValueError):
        return None


def _unwrap(d: dict | None) -> dict | None:
    """The Workflow tool persists an envelope ({summary, result, logs, ...}); the return is its result."""
    if isinstance(d, dict) and "eval_dir" not in d and isinstance(d.get("result"), dict):
        return d["result"]
    return d


def fold_overlay(ps_args: dict, kf_overlay: str) -> str:
    """The fused overlay REPLACES the initial one: the integrator built it as a copy of the initial
    overlay plus the fused kernels, and roles copy CURRENT_OVERLAY as one directory."""
    ps_args["initial_overlay_pythonpath"] = kf_overlay
    return kf_overlay


def run_stage(h: dict, ps_args: dict, timeout_s: int, G: dict, flags=frozenset()) -> dict | None:
    """Run the stage if switched on; fold an accepted overlay into ps_args. G is run_e2e's globals()
    (its invoke_workflow, PROCESS_SAFETY and the E2E_SCRIPT it launches). Returns the summary
    recorded under result.json["kernel_fusion"], or None when the switch is off. Never raises except
    for run_e2e's own SIGTERM self-stop, which must keep propagating."""
    cfg = config(h, flags)
    if not cfg:
        return None
    t0 = time.time()
    budget = stage_budget_s(cfg, timeout_s)
    args = build_args(cfg, ps_args, budget)
    kf_dir = Path(args["kf_dir"])
    summary: dict[str, Any] = {"schema": SCHEMA, "enabled": True, "kf_dir": str(kf_dir), "budget_s": budget,
                               "steps": args.get("kf_steps", "trace,topk,author,apply"),
                               "prompt": args.get("kf_prompt", "")}
    wf = None
    problems = validate(args)
    prior = (_read_json(kf_dir / "kf_return.json") or {}).get("workflow_return")
    if prior and prior.get("status") in ("accepted", "no_gain", "analysis_only"):
        wf = prior                  # a resumed run: the stage already finished for this eval dir
        summary["reused"] = True
    elif problems:
        summary.update(status="error", error="; ".join(problems))
    else:
        kf_dir.mkdir(parents=True, exist_ok=True)
        try:
            wf = _invoke(G, args, budget)
        except BaseException as e:  # noqa: BLE001 -- the stage must never take the main run down
            if isinstance(e, TimeoutError) and str(e).startswith("signal "):
                raise               # run_e2e's SIGTERM handler: flush and exit, do not start the main run
            if isinstance(e, (KeyboardInterrupt, SystemExit)):
                raise
            summary["error"] = f"{type(e).__name__}: {e}"[:2000]
            wf = _unwrap(_read_json(kf_dir / "workflow_return.json"))
    elapsed = int(time.time() - t0)
    if wf:
        # The analyst cannot write .md reports from inside the workflow; it returns them and they land here.
        for path, text in (wf.pop("analysis_files", None) or {}).items():
            if text and str(path).startswith(str(kf_dir.parent) + os.sep):   # the director may suffix the dir
                try:
                    Path(path).parent.mkdir(parents=True, exist_ok=True)
                    Path(path).write_text(text, encoding="utf-8")
                except OSError:
                    pass
        summary.update({k: wf.get(k) for k in (
            "status", "baseline_throughput_tok_s", "final_throughput_tok_s", "throughput_speedup",
            "final_overlay", "accepted", "rejected", "topk_json", "analysis_report", "trace_dir", "reason")})
    summary.setdefault("status", "error")
    summary["elapsed_s"] = elapsed
    if summary["status"] == "accepted" and summary.get("final_overlay"):
        summary["main_initial_overlay"] = fold_overlay(ps_args, summary["final_overlay"])
    remaining = max(int(timeout_s) - elapsed, 600)
    if "time_budget_s" in ps_args:
        ps_args["time_budget_s"] = remaining
    summary["main_timeout_s"] = remaining
    try:
        kf_dir.mkdir(parents=True, exist_ok=True)
        (kf_dir / "kf_return.json").write_text(json.dumps({"summary": summary, "workflow_return": wf}, indent=2))
    except OSError:
        pass
    return summary


def attach(result_path: Path, out: dict, kf: dict) -> dict:
    """Add the stage summary to the main result.json; combined speedup = stage A/B x main A/B."""
    out["kernel_fusion"] = kf
    try:
        out["kernel_fusion"]["combined_speedup_vs_original"] = (
            float(kf.get("throughput_speedup") or 1.0) * float(out.get("throughput_speedup") or 0.0)) or None
    except (TypeError, ValueError):
        pass
    try:
        tmp = result_path.with_name(result_path.name + ".tmp")
        tmp.write_text(json.dumps(out, indent=2), encoding="utf-8")
        os.replace(tmp, result_path)
    except OSError:
        pass
    return out


def dry_run_info(h: dict, ps_args: dict, timeout_s: int, flags=frozenset()) -> dict:
    cfg = config(h, flags)
    if not cfg:
        return {"enabled": False}
    args = build_args(cfg, ps_args, stage_budget_s(cfg, timeout_s))
    return {"enabled": True, "script": str(KF_SCRIPT), "args": args, "problems": validate(args)}
