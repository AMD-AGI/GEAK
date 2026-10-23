#!/usr/bin/env python3
"""Reviewed task_decls.json + the profile evidence -> task_plan.json, every task's
run/requested_kernel.json and case plan, and a scaffolded UT from its contract family.

    python3 build_task_plan.py --eval-dir "$EVAL_DIR" [--decls task_decls.json] [--check]

The agent's judgement lives in task_decls.json (draft_task_decls.py writes the first
version). Everything else a task needs is evidence already on disk, assembled here the
same way every run, so sibling tasks cannot disagree about a census list, a points file
or a denominator:

  task_plan.json                 tasks[] (selected_identities, companions, phases, points,
                                 family, target, case counts) and blocked_identities[]
                                 (excluded + blocked, each with its reason) -- the shape
                                 stages/profile/finalize.py reads
  tasks/<label>/run/requested_kernel.json   the declaration a UT lane works from
  tasks/<label>/run/ut_case_plan.json       plan_ut_cases.py's cases for the task
  tasks/<label>/run/server_variants.json   the Triton compile variants the probe server
                                 built for the task's kernels (evidence/compile_variants.json,
                                 from process start: warmup and graph capture included)
  tasks/<label>/run/server_calls.json      the entry calls that built them, with their
                                 metadata tensors under run/server_calls/ -- replayed by the
                                 UT's server_calls gate; meta.server_evidence points at both
  tasks/<label>/...              `ut_contracts/scaffold_ut.py new` with the family, target,
                                 const axes/attrs, phase regimes and the case manifest
                                 (when the family is not `custom` and case_axes is set;
                                 an existing unittest.py is never overwritten without --force)

meta.server_preimports (decls `server_preimports`, default by framework: sglang ->
sglang.srt.utils.common) are imported by both UT legs before the task's cases, so the kernel
compiles in the import environment the server has.

Lint (also `--check` alone, no writes): every selected identity owned exactly once by a
task, an exclusion or a blocker; only selected identities as task kernels (others are
companions, each with a reason); a known family or `custom` with a reason; joint rows
whose kernels belong to the task and launched at every point they name; unique labels;
and nothing left in any `needs_review`.
Stdlib only.
"""
import argparse
import json
import os
import re
import shutil
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "shared"))
import compile_watch  # noqa: E402
from plan_ut_cases import plan as plan_cases  # noqa: E402

SKILL = os.path.dirname(HERE)
SCAFFOLD = os.path.join(SKILL, "ut_contracts", "scaffold_ut.py")
FAMILIES = os.path.join(SKILL, "ut_contracts", "families")
LABEL_RE = re.compile(r"^[A-Za-z0-9_.-]+$")
# Modules the serving process has imported before any kernel compiles, whose import-time
# rebinding of library functions the UT must reproduce (sglang.srt.utils.common replaces
# triton.next_power_of_2). Light on purpose: every timing bucket is a fresh process.
SERVER_PREIMPORTS = {"sglang": ["sglang.srt.utils.common"]}


def load(path, default=None):
    try:
        with open(path) as f:
            return json.load(f)
    except (OSError, ValueError):
        return default


def dump(obj, path, indent=1):
    with open(path, "w") as f:
        json.dump(obj, f, indent=indent)


def launched_at(shapes_dir):
    out = {}
    if os.path.isdir(shapes_dir):
        for f in os.listdir(shapes_dir):
            if f.endswith(".json"):
                d = load(os.path.join(shapes_dir, f), {})
                out[d.get("point_id") or f[:-5]] = set(d.get("kernels", {}))
    return out


def unfilled_axes(t):
    """What the family contract still needs before the UT can be scaffolded."""
    fam = load(os.path.join(FAMILIES, t["ut_family"] + ".json"), {})
    axes = fam.get("axes") or {}
    var = [n for n, a in axes.items() if a.get("var")]
    const = [n for n, a in axes.items() if not a.get("var") and "expr" not in a]
    out = []
    miss = [n for n in const if n not in (t.get("axes") or {})]
    if miss:
        out.append("%s: axes still to pin for family %s: %s" % (t.get("label"), t["ut_family"], miss))
    for ph in t.get("phases") or []:
        have = (t.get("case_axes") or {}).get(ph) or {}
        miss = [n for n in var if n not in have]
        if miss:
            out.append("%s: case_axes.%s still needs %s (expressions in `shape`, `context`)"
                       % (t.get("label"), ph, miss))
    return out


def lint(decls, targets, launched, accept_draft=False):
    errs, warns = [], []
    known = {f[:-5] for f in os.listdir(FAMILIES) if f.endswith(".json")}
    owner = {}

    def own(k, who):
        if k in owner:
            errs.append("%s is owned twice: %s and %s" % (k[:80], owner[k], who))
        owner[k] = who
    labels = set()
    for t in decls.get("tasks", []):
        lab = t.get("label") or ""
        if not LABEL_RE.match(lab) or lab in labels:
            errs.append("task label %r is empty, not a plain directory name, or repeated" % lab)
        labels.add(lab)
        for k in t.get("kernels") or []:
            if k not in targets:
                errs.append("%s: %s is not a selected identity; list it under companions "
                            "with a reason" % (lab, k[:80]))
            own(k, "task " + lab)
        if not t.get("kernels"):
            errs.append("%s owns no kernel" % lab)
        for c in t.get("companions") or []:
            if not (c.get("kernel") and (c.get("why") or "").strip()):
                errs.append("%s: every companion needs kernel and why" % lab)
        fam = t.get("ut_family")
        if fam == "custom":
            if not (t.get("ut_family_why") or "").strip():
                errs.append("%s: ut_family custom needs ut_family_why" % lab)
        elif fam not in known:
            errs.append("%s: unknown ut_family %r (scaffold_ut.py list)" % (lab, fam))
        if fam in known:
            warns += unfilled_axes(t)
        if not t.get("target_callable"):
            warns.append("%s: no target_callable; the UT scaffold will carry a FILL marker" % lab)
        members = set(t.get("kernels") or []) | {c.get("kernel") for c in t.get("companions") or []}
        for j in t.get("joint_phases") or []:
            extra = set(j.get("required_kernels") or []) - members
            if extra:
                errs.append("%s/%s requires kernels outside the task: %s"
                            % (lab, j.get("regime"), sorted(x[:60] for x in extra)))
            for p in j.get("points") or []:
                miss = [k for k in j.get("required_kernels") or [] if k not in launched.get(p, set())]
                if p in launched and miss:
                    errs.append("%s/%s: %s did not launch %s; mutually exclusive branches are "
                                "separate rows" % (lab, j.get("regime"), p, [m[:50] for m in miss]))
        if t.get("needs_review") and not accept_draft:
            errs.append("%s: needs_review is not empty (%d item(s))" % (lab, len(t["needs_review"])))
    for kind in ("excluded", "blocked"):
        for x in decls.get(kind, []):
            why = (x.get("reason") or x.get("blocker") or "").strip()
            if not why:
                errs.append("%s entry for %s has no reason" % (kind, x.get("kernels")))
            for k in x.get("kernels") or []:
                own(k, kind)
            if x.get("needs_review") and not accept_draft:
                errs.append("%s entry %s: needs_review is not empty" % (kind, x.get("kernels")))
    for k in targets:
        if k not in owner:
            errs.append("selected identity has no task, exclusion or blocker: %s" % k[:100])
    return errs, warns


def manifest(task, cases):
    """ut_case_plan cases -> the contract case manifest scaffold_ut.py consumes."""
    out, by_phase = [], {}
    for c in cases:
        expr = (task.get("case_axes") or {}).get(c["phase"])
        if not expr:
            return None
        names = {"shape": c["shape"], "context": c.get("context") or c["shape"]}
        axes = {ax: int(round(eval(e, {"__builtins__": {}}, names)))  # noqa: S307
                for ax, e in expr.items()}
        sig = "%s_%s%s" % (c["phase"], c["shape"], "" if c["kind"] == "point" else "_" + c["kind"])
        out.append({"sig": sig, "regime": c["phase"], "axes": axes,
                    "effective_calls": c["effective_calls"], "timed": c["kind"] != "envelope",
                    "random": True, "kind": c["kind"],
                    "evidence": c.get("shape_manifest") or "ut_case_plan.json#" + c["case_id"]})
        by_phase.setdefault(c["phase"], []).append((c["shape"], sig))
    replay = []
    for ph, rows in by_phase.items():
        if (task.get("phase_regimes") or {}).get(ph, {}).get("cuda_graph"):
            rows.sort()
            replay += sorted({rows[0][1], rows[-1][1]})
    return {"cases": out, "replay_sigs": replay}


def joint_ut(t, rank):
    """task_decls joint_phases -> delivery-contract joint_ut.phases[] rows (anchor = the
    row's heaviest kernel; a row of one kernel is not a chain and is reported apart)."""
    seam = [{"target_callable": t.get("target_callable"), "path": t.get("target_file"),
             "sha256": t.get("target_sha256"), "role": "op_seam",
             "evidence": "innermost serving-stack frame common to the task's kernels "
                         "(evidence/source_map.json)"}] if t.get("target_callable") else []
    rows, single = [], []
    js = t.get("joint_phases") or []
    branches = {k for j in js for k in j.get("required_kernels") or []}
    if js and all(len(j.get("required_kernels") or []) == 1 for j in js) and len(branches) > 1:
        # Every row one kernel, the kernels distinct: mutually exclusive branches of one
        # whole op, which the delivery contract declares as dispatch_ut, never joint_ut.
        return None, [], {"phases": [
            {"regime": j["regime"], "device_kernel": j["required_kernels"][0],
             "served_phase": j.get("phase"), "owned_point_ids": j.get("points"),
             "seam_candidates": seam} for j in js]}
    for j in js:
        ks = list(dict.fromkeys(j.get("required_kernels") or []))
        if len(ks) < 2:
            single.append(j)
            continue
        rows.append({"regime": j["regime"],
                     "device_kernel": max(ks, key=lambda k: rank.get(k, {}).get("mass_us", 0)),
                     "required_device_kernels": ks, "served_phase": j.get("phase"),
                     "owned_point_ids": j.get("points"), "seam_candidates": seam})
    return ({"phases": rows} if rows else None), single, None


def frameworks(task, cv):
    """Serving frameworks on the task's launch paths (its entry or any server stack)."""
    mods = [(task.get("target_callable") or "").split(".")[0]]
    for r in (cv or {}).get("variants", []):
        mods += [f.split(".")[0] for f in r.get("stack") or []]
    return sorted({m for m in mods if m in SERVER_PREIMPORTS})


def server_evidence(eval_dir, task, tdir):
    """The task's slice of evidence/compile_variants.json -> run/server_variants.json and
    run/server_calls.json (+ run/server_calls/<variant>.pt). Returns the meta block."""
    cv = load(os.path.join(eval_dir, "evidence", "compile_variants.json"))
    pre = task.get("server_preimports")
    if pre is None:
        pre = [m for f in frameworks(task, cv) for m in SERVER_PREIMPORTS[f]]
    if not cv:
        return {"server_preimports": pre, "server_evidence": None,
                "why": "no evidence/compile_variants.json (probe server ran without compile_watch)"}
    names = set(task.get("kernels") or []) | {c.get("kernel") for c in task.get("companions") or []}
    target = task.get("target_callable")
    recs = [r for r in cv["variants"] if compile_watch.belongs(r, names, target)]
    slim = [{k: v for k, v in r.items() if k not in ("frames", "tensors")} for r in recs]
    run = os.path.join(tdir, "run")
    shutil.copy2(compile_watch.__file__, os.path.join(tdir, "compile_watch.py"))
    dump({"contract": "kernel_extract/server_variants/v1",
               "source": os.path.join(eval_dir, "evidence", "compile_variants.json"),
               "server_mode": cv.get("server_mode"), "env": cv.get("env") or {},
               "kernels": sorted(names), "target_callable": target, "variants": slim},
         os.path.join(run, "server_variants.json"))
    calls, unmatched, seen = [], [], set()
    shutil.rmtree(os.path.join(run, "server_calls"), ignore_errors=True)
    for r in recs:
        fr = compile_watch.frame_for(r, target) if target else None
        if fr is None:
            if r.get("frames"):
                unmatched.append({"variant": r["id"], "kernel": r["name"],
                                  "frames": [f["callable"] for f in r["frames"]]})
            continue
        key = json.dumps(fr["args"], sort_keys=True, default=str)
        if key in seen:
            for c in calls:
                if c["_key"] == key:
                    c["variants"].append(r["id"])
            continue
        seen.add(key)
        rel = None
        if r.get("tensors"):
            src = os.path.join(cv["watch_dir"], r["tensors"])
            if os.path.exists(src):
                os.makedirs(os.path.join(run, "server_calls"), exist_ok=True)
                rel = os.path.join("run", "server_calls", r["id"] + ".pt")
                shutil.copy2(src, os.path.join(tdir, rel))
        calls.append({"_key": key, "id": r["id"], "variants": [r["id"]], "kernel": r["name"],
                      "callable": fr["callable"], "file": fr.get("file"), "line": fr.get("line"),
                      "positional": fr["positional"], "args": fr["args"], "tensors": rel,
                      "in_graph_capture": bool(r.get("in_graph_capture") and not r.get("eager")),
                      "before_ready": r.get("before_ready")})
    for c in calls:
        c.pop("_key")
    dump({"contract": "kernel_extract/server_calls/v1", "target_callable": target,
          "calls": calls, "unmatched_variants": unmatched}, os.path.join(run, "server_calls.json"))
    return {"server_preimports": pre,
            "server_evidence": {"compile_variants": "run/server_variants.json",
                                "server_calls": "run/server_calls.json",
                                "server_mode": cv.get("server_mode"),
                                "variants": len(slim), "calls": len(calls),
                                "variants_without_entry_frame": len(unmatched)}}


def stamp_meta(tdir, ev, waivers):
    """Write the server evidence into an existing meta.json (scaffolded now or before)."""
    path = os.path.join(tdir, "meta.json")
    meta = load(path)
    if meta is None:
        return False
    meta["server_preimports"] = ev["server_preimports"]
    if ev.get("server_evidence"):
        meta["server_evidence"] = ev["server_evidence"]
    if waivers is not None:
        meta.setdefault("ut_contract", {})["compile_variant_waivers"] = waivers
    dump(meta, path, indent=2)
    return True


def scaffold(task, tdir, man, force):
    if task.get("ut_family") in (None, "custom"):
        return {"status": "skipped", "why": "custom family: write a contract (--family-file)"}
    if man is None:
        return {"status": "skipped", "why": "case_axes not set for every owned phase"}
    if os.path.exists(os.path.join(tdir, "unittest.py")) and not force:
        return {"status": "kept", "why": "unittest.py exists; --force to regenerate"}
    mpath = os.path.join(tdir, "run", "ut_cases.manifest.json")
    rpath = os.path.join(tdir, "run", "phase_regimes.json")
    json.dump(man, open(mpath, "w"), indent=1)
    json.dump(task.get("phase_regimes") or {}, open(rpath, "w"), indent=1)
    cmd = [sys.executable, SCAFFOLD, "new", "--family", task["ut_family"], "--task", tdir,
           "--manifest", mpath, "--phase-regimes", rpath]
    if task.get("target_callable"):
        cmd += ["--target", task["target_callable"]]
    for k, v in (task.get("axes") or {}).items():
        cmd += ["--axis", "%s=%s" % (k, v)]
    for k, v in (task.get("attrs") or {}).items():
        cmd += ["--attr", "%s=%s" % (k, v)]
    if force:
        cmd.append("--force")
    p = subprocess.run(cmd, capture_output=True, text=True)
    lines = [l for l in (p.stdout + p.stderr).splitlines() if l.strip()]
    return {"status": "ok" if p.returncode == 0 else "failed", "rc": p.returncode,
            "error": lines[-1] if p.returncode and lines else None,
            "command": cmd, "output": "\n".join(lines[-30:])}


def build(eval_dir, decls, accept_draft=False, check_only=False, force=False, no_scaffold=False):
    ev = lambda *p: os.path.join(eval_dir, *p)  # noqa: E731
    sel = load(ev("selection_targets.json"), {})
    targets = [t["id"] for t in sel.get("targets", [])]
    shapes_dir = ev("evidence", "shapes")
    launched = launched_at(shapes_dir)
    errs, warns = lint(decls, targets, launched, accept_draft)
    if errs or check_only:
        return {"errors": errs, "warnings": warns}

    points = load(ev("sampling_points_final.json"))
    if not points:
        raise SystemExit("build_task_plan: sampling_points_final.json is missing (relabel_points.py)")
    cases = plan_cases(points, shapes_dir, decls["tasks"])["tasks"]
    census = [{"point_id": p["point_id"], "phase": p["phase"], "path": p.get("census"),
               "achieved_shape": p.get("achieved_shape"),
               "instances_represented": p["represented"]["instances"]}
              for p in points["points"] if p.get("counted")]
    shared = {"eval_dir": os.path.abspath(eval_dir),
              "kernel_ranking": ev("kernel_ranking.json"),
              "selection_targets": ev("selection_targets.json"),
              "sampling_points": ev("sampling_points_final.json"),
              "coverage": ev("coverage.json"), "source_map": ev("evidence", "source_map.json"),
              "adjacency": ev("evidence", "adjacency.json"), "shapes_dir": shapes_dir,
              "census": census,
              "census_rule": "grade coverage with exactly these census pairs and this points "
                             "file; a partial list shrinks the denominator"}
    rank = {r["id"]: r for r in load(ev("kernel_ranking.json"), {}).get("ranking", [])}
    tasks = []
    for t in decls["tasks"]:
        tdir = ev("tasks", t["label"])
        os.makedirs(os.path.join(tdir, "run"), exist_ok=True)
        tc = cases.get(t["label"], {"cases": []})
        owned = sorted({c["point_id"] for c in tc["cases"] if c.get("point_id")})
        req = {
            "contract": "kernel_extract/requested_kernel/v1",
            "label": t["label"], "task_dir": tdir,
            "selected_identities": t["kernels"], "selected_device_kernels": t["kernels"],
            "device_kernel": max(t["kernels"], key=lambda k: rank.get(k, {}).get("mass_us", 0)),
            "required_companions": t.get("companions") or [],
            "phases": t.get("phases"), "owned_point_ids": owned,
            "ut_family": t["ut_family"], "ut_family_why": t.get("ut_family_why"),
            "target_callable": t.get("target_callable"), "target_file": t.get("target_file"),
            "target_sha256": t.get("target_sha256"), "launching_ops": t.get("launching_ops"),
            "joint_ut": joint_ut(t, rank)[0],
            "single_kernel_regimes": joint_ut(t, rank)[1] or None,
            "dispatch_ut": joint_ut(t, rank)[2],
            "phase_regimes": t.get("phase_regimes"),
            "selection_pct_gpu_time": round(sum(rank.get(k, {}).get("selection_pct_gpu_time", 0)
                                                for k in t["kernels"]), 4),
            "source_delivery_required": True, "required_gpu_count": 1,
            "capture_required": False,
            "shared": shared,
            "rules": ["hold one card; start no server, take no capture -- if evidence is "
                      "missing, return blocked naming it",
                      "write only inside task_dir"],
        }
        json.dump(req, open(os.path.join(tdir, "run", "requested_kernel.json"), "w"), indent=1)
        json.dump(tc, open(os.path.join(tdir, "run", "ut_case_plan.json"), "w"), indent=1)
        se = server_evidence(eval_dir, t, tdir)
        sc = {"status": "skipped", "why": "--no-scaffold"} if no_scaffold else \
            scaffold(t, tdir, manifest(t, tc["cases"]), force)
        stamp_meta(tdir, se, t.get("compile_variant_waivers"))
        req["server_preimports"] = se["server_preimports"]
        req["server_evidence"] = se.get("server_evidence") or {"missing": se.get("why")}
        json.dump(req, open(os.path.join(tdir, "run", "requested_kernel.json"), "w"), indent=1)
        kinds = {}
        for c in tc["cases"]:
            kinds[c["kind"]] = kinds.get(c["kind"], 0) + 1
        tasks.append({"label": t["label"], "task_dir": tdir,
                      "selected_identities": t["kernels"],
                      "required_companions": [c["kernel"] for c in t.get("companions") or []],
                      "phases": t.get("phases"), "owned_point_ids": owned,
                      "ut_family": t["ut_family"], "target_callable": t.get("target_callable"),
                      "planned_cases": dict(kinds, total=len(tc["cases"])),
                      "proxy_mass_outside_every_case_pct": tc.get("proxy_mass_outside_every_case_pct"),
                      "selection_pct_gpu_time": req["selection_pct_gpu_time"],
                      "requested_kernel": os.path.join(tdir, "run", "requested_kernel.json"),
                      "server_evidence": req["server_evidence"],
                      "scaffold": sc})
    if any("missing" in (t["server_evidence"] or {}) for t in tasks):
        warns.append("no evidence/compile_variants.json: the UTs get no server_calls or "
                     "compile_variants gate (run the probe through probe_server.py, which "
                     "records them by default)")
    blocked = []
    for kind in ("excluded", "blocked"):
        for x in decls.get(kind, []):
            for k in x["kernels"]:
                blocked.append({"kernel": k, "kind": x.get("kind", kind),
                                "blocker": x.get("reason") or x.get("blocker")})
    plan = {"contract": "agentx_task_plan/v1", "scope": "UT generation; built by "
            "build_task_plan.py from task_decls.json", "decls": decls.get("_path"),
            "shared": shared, "tasks": tasks, "blocked_identities": blocked,
            "selected_overall_pct": round(sum(t.get("overall_pct") or 0
                                              for t in sel.get("targets", [])), 4),
            "warnings": warns}
    json.dump(plan, open(ev("task_plan.json"), "w"), indent=1)
    return {"errors": [], "warnings": warns, "plan": plan}


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--eval-dir", required=True)
    ap.add_argument("--decls", default=None, help="default: <eval-dir>/task_decls.json")
    ap.add_argument("--check", action="store_true", help="lint only, write nothing")
    ap.add_argument("--accept-draft", action="store_true",
                    help="ignore needs_review (tests and dry runs only)")
    ap.add_argument("--no-scaffold", action="store_true")
    ap.add_argument("--force", action="store_true", help="regenerate existing UT scaffolds")
    a = ap.parse_args()
    path = a.decls or os.path.join(a.eval_dir, "task_decls.json")
    decls = load(path)
    if not decls:
        raise SystemExit("build_task_plan: no declaration at %s (draft_task_decls.py)" % path)
    decls["_path"] = os.path.abspath(path)
    res = build(a.eval_dir, decls, a.accept_draft, a.check, a.force, a.no_scaffold)
    for w in res["warnings"]:
        print("  warning: %s" % w)
    if res["errors"]:
        for e in res["errors"]:
            print("  ERROR: %s" % e)
        print("build_task_plan: %d error(s); nothing written" % len(res["errors"]))
        return 1
    if a.check:
        print("build_task_plan: declaration is consistent")
        return 0
    for t in res["plan"]["tasks"]:
        sc = t["scaffold"]
        print("  %-44s %-18s cases %-32s scaffold %s%s"
              % (t["label"], t["ut_family"], json.dumps(t["planned_cases"]), sc["status"],
                 ": " + (sc.get("error") or sc.get("why") or "") if sc["status"] != "ok" else ""))
    print("-> %s" % os.path.join(a.eval_dir, "task_plan.json"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
