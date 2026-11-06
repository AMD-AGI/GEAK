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

import hashlib
import json
import os
import re
import time
from pathlib import Path
from typing import Any

KF_HOME = Path(__file__).resolve().parent
KF_SCRIPT = KF_HOME / "kf_workflow.js"
DEFAULT_BUDGET_S = 12 * 3600
# The stage may hold at most this share of the run, hard stop included; the main workflow keeps the rest.
MAX_RUN_SHARE = 0.4
# Below this the stage cannot fit a baseline, a trace, the agents and one A/B: it is skipped, not squeezed.
MIN_BUDGET_S = 2 * 3600
SCHEMA = "geak.kernel_fusion/1"

# handoff key / env var -> kf_workflow.js arg
# The fusion-analysis skill every analysis follows (fusion levels, requirements, procedure, estimation, report
# contents). Run-specific prompts (deployment, workload, extra constraints) are read alongside it.
FUSION_STRATEGY = KF_HOME.parent / "knowledge" / "kernel_fusion_skills" / "fusion-analysis" / "SKILL.md"

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
    "baseline_mode": ("GEAK_KF_BASELINE_MODE", "kf_baseline_mode"),
    "implementable_levels": ("GEAK_KF_IMPLEMENTABLE_LEVELS", "kf_implementable_levels"),
    "unsupported_patterns": ("GEAK_KF_UNSUPPORTED_PATTERNS", "kf_unsupported_patterns"),
    "accuracy_confirm_limit": ("GEAK_KF_ACCURACY_CONFIRM_LIMIT", "kf_accuracy_confirm_limit"),
    "min_gain_pct": ("GEAK_KF_MIN_GAIN_PCT", "kf_min_gain_pct"),
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


def hard_timeout_s(budget_s: int) -> int:
    """The stage's hard stop: the JS stops starting work at 85% of its budget; the slack lets in-flight legs land."""
    return int(budget_s * 1.1) + 900


def stage_budget_s(cfg: dict, timeout_s: int) -> int:
    """min(12 h, 1/3 of the run), or the configured budget; either way capped so the stage's HARD stop stays
    within MAX_RUN_SHARE of the run."""
    budget = int(cfg["budget_s"]) if cfg.get("budget_s") else int(min(DEFAULT_BUDGET_S, max(timeout_s // 3, 1)))
    cap = int((MAX_RUN_SHARE * int(timeout_s) - 900) / 1.1)
    return max(min(budget, cap), 0)


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
        a["kf_topk"] = slim_topk(d, int(a.get("kf_author_k") or 0))
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
    """Every candidate (the Feasibility Review judges them all); with an author_k cap, only the first author_k
    recommended ones plus every candidate they name (fallback / exclusive)."""
    cands = {str(c.get("id")): c for c in d.get("candidates") or [] if isinstance(c, dict)}
    rec = [str(i) for i in d.get("recommended") or []]
    if author_k <= 0:
        return {"candidates": [{k: c[k] for k in CAND_KEYS if k in c} for c in cands.values()], "recommended": rec}
    rec = rec[:author_k]
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
        f'returns, write its full return value to "{ret}" with the Write tool (it is large: never print it), '
        "then print EXACTLY ONE final line of compact JSON, this pointer and nothing else: "
        + json.dumps({"eval_dir": args["kf_dir"], "workflow_return_json": ret, "status": "<the return's status>"})
        + "\nPrint nothing after that line. If the Workflow tool fails, do NOT edit any file, fix the script or "
        "launch it again: print the error and stop (the caller records it and continues)."
    )


def _invoke(G: dict, args: dict, budget_s: int) -> dict:
    saved = G["E2E_SCRIPT"]
    G["E2E_SCRIPT"] = KF_SCRIPT     # the standalone-runtime path launches run_e2e's global E2E_SCRIPT
    try:
        # The JS stops starting work at 85% of time_budget_s; the slack lets in-flight A/B legs land.
        return G["invoke_workflow"](build_prompt(G, args), hard_timeout_s(budget_s), args["kf_dir"], ps_args=args)
    finally:
        G["E2E_SCRIPT"] = saved


def stop_leftovers(G: dict, kf_dir: Path) -> dict:
    """Stop what the stage left running, however it ended: a leg launched with ``setsid nohup`` outlives the
    workflow and every agent in it, and would hold the serving GPUs (and their lock) through the main run.

    Uses GEAK's own run reaper (scripts/run_reaper.py, read-only) with the stage dir as the only root: a
    process is the stage's when it started after this runner did and its command line or working directory
    is inside ``kf_dir`` -- plus everything under it. The runner and its ancestors are never signalled.
    """
    loader = G.get("_run_reaper")
    if not callable(loader):
        return {"note": "no run reaper in this runner; nothing stopped"}
    try:
        R = loader()
        roots = R.eval_dir_roots(str(kf_dir))
        if not roots:
            return {"note": f"{kf_dir} is too shallow to match on; nothing stopped"}
        chain = R.protected_pids()
        st = R._stat(os.getpid())
        procs = R.snapshot("", roots, st[1] if st else None)
        chosen = R.select(procs, "exit", set(chain))
        if not chosen:
            return {"stopped": [], "survivors": []}
        stopped, survivors = R.stop(chosen, term_grace_s=R.TERM_GRACE_S, kill_wait_s=R.KILL_WAIT_S)
        return {"stopped": stopped, "survivors": survivors}
    except Exception as e:  # noqa: BLE001 -- a cleanup fault must not take the main run down
        return {"error": f"{type(e).__name__}: {e}"[:500]}


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


# Benchmark / correctness output the integrator writes next to the overlay sources; not part of the overlay.
_OVERLAY_OUTPUT_DIRS = {"__pycache__", "cand", "ref", "correctness"}
_ABS_PATH = re.compile(r"""["'](/[^"'\s]+)["']""")


def _overlay_files(root: Path) -> dict[str, Path]:
    out: dict[str, Path] = {}
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [d for d in dirnames if d not in _OVERLAY_OUTPUT_DIRS]
        for f in filenames:
            if f.endswith((".pyc", ".log", ".out")) or f == "integrate_result.json":
                continue
            p = Path(dirpath) / f
            out[str(p.relative_to(root))] = p
    return out


def _ab_record(overlay: Path, kf_dir: Path) -> tuple[dict | None, Path | None]:
    """The integrate_result.json whose A/B accepted exactly this overlay directory."""
    for p in sorted(kf_dir.rglob("integrate_result.json")):
        d = _read_json(p) or {}
        acc = d.get("accepted_overlay")
        if acc and Path(acc).resolve() == overlay:
            return d, p
    return None, None


def validate_overlay(kf_overlay: str, initial_overlay: str, kf_dir: Path) -> dict:
    """Checks the stage's accepted overlay before it replaces the main workflow's initial overlay.

    1. it is a directory inside the stage dir with a sitecustomize.py;
    2. every .py in it compiles (compiled only, never imported or run here);
    3. every absolute path it names under the stage dir (the fused kernels it loads) exists;
    4. existing optimizations are kept: every file of the initial overlay is present and byte-identical,
       and the initial sitecustomize.py is either kept as is or copied under another name that the new
       sitecustomize.py imports;
    5. it is the overlay the A/B measured: an integrate_result.json accepted exactly this directory with a
       complete A/B, and no overlay source changed after that A/B's candidate leg finished.
    Returns {"ok", "problems", "manifest_sha256", "files"}."""
    problems: list[str] = []
    kf_dir = Path(kf_dir).resolve()
    overlay = Path(kf_overlay).resolve() if kf_overlay else Path()
    if not kf_overlay or not overlay.is_dir():
        return {"ok": False, "problems": [f"overlay {kf_overlay!r} is not a directory"]}
    if kf_dir not in overlay.parents:
        problems.append(f"overlay {overlay} is outside the stage dir {kf_dir}")
    files = _overlay_files(overlay)
    site = overlay / "sitecustomize.py"
    if not site.is_file():
        problems.append("no sitecustomize.py at the overlay root")
    for rel, p in sorted(files.items()):
        if not rel.endswith(".py"):
            continue
        try:
            src = p.read_text(encoding="utf-8")
            compile(src, str(p), "exec")
        except (OSError, UnicodeDecodeError, SyntaxError, ValueError) as e:
            problems.append(f"{rel} does not compile: {type(e).__name__}: {e}"[:300])
            continue
        for ref in _ABS_PATH.findall(src):
            if ref.startswith(str(kf_dir.parent) + os.sep) and not Path(ref).exists():
                problems.append(f"{rel} names {ref}, which does not exist")
    site_src = site.read_text(encoding="utf-8", errors="replace") if site.is_file() else ""
    for root in [Path(x) for x in str(initial_overlay or "").split(os.pathsep) if x.strip()]:
        if not root.is_dir():
            continue
        for rel, p in sorted(_overlay_files(root).items()):
            want = p.read_bytes()
            if rel == "sitecustomize.py":
                if site.is_file() and site.read_bytes() == want:
                    continue
                kept = [r for r, q in files.items() if r != "sitecustomize.py" and q.read_bytes() == want]
                imported = [r for r in kept if re.search(
                    r"\b" + re.escape(Path(r).with_suffix("").as_posix().replace("/", ".")) + r"\b", site_src)]
                if not kept:
                    problems.append(f"the initial overlay's sitecustomize.py ({root}) is not kept")
                elif not imported:
                    problems.append(f"the initial overlay's sitecustomize.py is kept as {kept[0]} "
                                    "but the new sitecustomize.py does not import it")
            elif rel not in files:
                problems.append(f"initial overlay file {rel} ({root}) is missing")
            elif files[rel].read_bytes() != want:
                problems.append(f"initial overlay file {rel} ({root}) was changed")
    rec, rec_path = _ab_record(overlay, kf_dir)
    if rec is None:
        problems.append("no integrate_result.json accepted this overlay directory")
    elif not rec.get("ab_complete"):
        problems.append(f"{rec_path}: the A/B is not complete")
    else:
        leg = Path(((rec.get("measurement_evidence") or {}).get("cand_summary")) or "")
        stamp = (leg if leg.is_file() else rec_path).stat().st_mtime
        changed = [r for r, p in files.items() if p.stat().st_mtime > stamp + 2]
        if changed:
            problems.append(f"overlay sources changed after the A/B measured it: {', '.join(sorted(changed)[:5])}")
    h = hashlib.sha256()
    for rel, p in sorted(files.items()):
        h.update(rel.encode() + b"\0" + hashlib.sha256(p.read_bytes()).digest())
    return {"ok": not problems, "problems": problems, "manifest_sha256": h.hexdigest(), "files": len(files)}


def fold_overlay(ps_args: dict, kf_overlay: str) -> str:
    """The fused overlay REPLACES the initial one: the integrator built it as a copy of the initial
    overlay plus the fused kernels, and roles copy CURRENT_OVERLAY as one directory."""
    ps_args["initial_overlay_pythonpath"] = kf_overlay
    return kf_overlay


def _passes_through(e: BaseException) -> bool:
    """run_e2e's SIGTERM self-stop and an operator's Ctrl-C/exit must keep propagating; nothing else may."""
    return (isinstance(e, (KeyboardInterrupt, SystemExit))
            or (isinstance(e, TimeoutError) and str(e).startswith("signal ")))


def run_stage(h: dict, ps_args: dict, timeout_s: int, G: dict, flags=frozenset()) -> dict | None:
    """Run the stage if switched on; fold an accepted overlay into ps_args. G is run_e2e's globals()
    (its invoke_workflow, PROCESS_SAFETY and the E2E_SCRIPT it launches). Returns the summary
    recorded under result.json["kernel_fusion"], or None when the switch is off.

    A failure anywhere in the stage -- config, args, the workflow, the leftover reaper, the overlay check,
    writing its record -- ends the stage as status "error": the main workflow then runs with its args exactly
    as they were and the run budget minus the seconds the stage used. Never raises except for run_e2e's own
    SIGTERM self-stop (and Ctrl-C/exit), which must keep propagating."""
    t0 = time.time()
    snapshot = dict(ps_args)
    try:
        return _run_stage(h, ps_args, timeout_s, G, flags, t0)
    except BaseException as e:
        if _passes_through(e):
            raise
        ps_args.clear()
        ps_args.update(snapshot)          # whatever the stage had folded in is undone
        elapsed = int(time.time() - t0)
        remaining = max(int(timeout_s) - elapsed, 60)
        if "time_budget_s" in ps_args:
            ps_args["time_budget_s"] = remaining
        return {"schema": SCHEMA, "enabled": True, "status": "error",
                "error": f"stage driver: {type(e).__name__}: {e}"[:2000],
                "elapsed_s": elapsed, "main_timeout_s": remaining}


def _run_stage(h: dict, ps_args: dict, timeout_s: int, G: dict, flags, t0: float) -> dict | None:
    cfg = config(h, flags)
    if not cfg:
        return None
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
    elif budget < MIN_BUDGET_S:
        summary.update(status="skipped", reason=(
            f"stage budget {budget}s < {MIN_BUDGET_S}s: the run ({int(timeout_s)}s) is too short to fit a baseline, "
            f"a trace and an A/B within {int(MAX_RUN_SHARE * 100)}% of it"))
    else:
        kf_dir.mkdir(parents=True, exist_ok=True)
        try:
            wf = _invoke(G, args, budget)
            if not isinstance(wf, dict) or "workflow_return_json" in wf or "status" not in wf:
                # The driver prints a pointer line; the return itself lives in the file it wrote. (Echoing a
                # 40 KB return as one stdout line was the step that failed: WorkflowParseError.)
                wf = _unwrap(_read_json(kf_dir / "workflow_return.json")) or wf
        except BaseException as e:  # the stage must never take the main run down
            if _passes_through(e):
                raise               # run_e2e's SIGTERM handler: flush and exit, do not start the main run
            wf = _unwrap(_read_json(kf_dir / "workflow_return.json"))
            if type(e).__name__ == "WorkflowParseError" and isinstance(wf, dict) and wf.get("status"):
                # The driver wrote the full return to disk and its session ended before the pointer line reached
                # stdout: the stage finished, nothing failed. Keep a note, not an error.
                summary["note"] = "return read from workflow_return.json (no pointer line on stdout)"
            else:
                summary["error"] = f"{type(e).__name__}: {e}"[:2000]
        finally:
            summary["leftovers"] = stop_leftovers(G, kf_dir)
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
        check = validate_overlay(summary["final_overlay"], ps_args.get("initial_overlay_pythonpath") or "", kf_dir)
        summary["overlay_check"] = check
        if check["ok"]:
            summary["main_initial_overlay"] = fold_overlay(ps_args, summary["final_overlay"])
        else:
            # The A/B accepted it, but it cannot be handed on as is: the main workflow keeps its own overlay.
            summary["status"] = "rejected_overlay"
            summary["reason"] = "accepted overlay failed validation: " + "; ".join(check["problems"])[:1500]
    elapsed = summary["elapsed_s"] = int(time.time() - t0)
    remaining = max(int(timeout_s) - elapsed, 60)
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
    budget = stage_budget_s(cfg, timeout_s)
    args = build_args(cfg, ps_args, budget)
    return {"enabled": True, "script": str(KF_SCRIPT), "args": args, "problems": validate(args),
            "budget_s": budget, "hard_timeout_s": hard_timeout_s(budget), "skipped": budget < MIN_BUDGET_S}
