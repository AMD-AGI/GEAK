#!/usr/bin/env python3
"""ut_request.json -> one task directory per requested task, scaffolded from its contract
family, plus task_plan.json. The UT stage's only input is the request: it reads no profile
evidence and knows nothing about where the cases or their weights came from.

    python3 scaffold_tasks.py --request <dir>/ut_request.json --out "$OUT" [--force] [--check]

For every task:
  tasks/<label>/run/            the request's files for it (ut_case_plan.json,
                                server_variants.json, server_calls.json, server_calls/*.pt)
  tasks/<label>/run/requested_kernel.json   the request's KERNEL object, with task_dir and
                                the server evidence installed here
  tasks/<label>/...             `ut_contracts/scaffold_ut.py new` with the family, target,
                                const axes/attrs, phase regimes and the case manifest
                                (an existing unittest.py is never overwritten without --force)
  tasks/<label>/meta.json       server_preimports, server_evidence, expected_launches and
                                compile_variant_waivers stamped from the request
  task_plan.json                tasks[] and blocked_identities[], the shape downstream reads
  selection_targets.json        carried from the request when it has one

`--check` verifies the request (contract, every listed file's SHA-256, task fields) and
writes nothing. Stdlib only.
"""
import argparse
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
SKILL = os.path.dirname(HERE)
STAGES = os.path.dirname(os.path.dirname(SKILL))
SCAFFOLD = os.path.join(SKILL, "ut_contracts", "scaffold_ut.py")
FAMILIES = os.path.join(SKILL, "ut_contracts", "families")
COMPILE_WATCH = os.path.join(STAGES, "_shared", "compile_watch.py")
CONTRACT = "workload/ut_request/v1"
LABEL_RE = re.compile(r"^[A-Za-z0-9_.-]+$")


def load(path, default=None):
    try:
        with open(path) as f:
            return json.load(f)
    except (OSError, ValueError):
        return default


def dump(obj, path, indent=1):
    with open(path, "w") as f:
        json.dump(obj, f, indent=indent)


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def check_request(path):
    """(request, its directory, problems)."""
    req = load(path)
    if not isinstance(req, dict) or req.get("contract") != CONTRACT:
        return None, None, ["%s is not a %s request" % (path, CONTRACT)]
    root = os.path.dirname(os.path.abspath(path))
    problems = []
    for rel, ref in (req.get("files") or {}).items():
        full = os.path.join(root, rel)
        if os.path.isabs(rel) or ".." in rel.split(os.sep):
            problems.append("file path %r is not inside the request directory" % rel)
        elif not os.path.isfile(full) or sha256(full) != ref.get("sha256"):
            problems.append("request file changed or is missing: %s" % rel)
    known = {f[:-5] for f in os.listdir(FAMILIES) if f.endswith(".json")}
    labels = set()
    for t in req.get("tasks") or []:
        lab = t.get("label") or ""
        if not LABEL_RE.match(lab) or lab in labels:
            problems.append("task label %r is empty, not a plain directory name, or repeated" % lab)
        labels.add(lab)
        sc = t.get("scaffold") or {}
        if sc.get("family") not in known | {"custom"} and not sc.get("family_file"):
            problems.append("%s: unknown family %r" % (lab, sc.get("family")))
        rk = t.get("requested_kernel") or {}
        if rk.get("entity_kind") != "gpu_kernel" or not rk.get("selected_identities"):
            problems.append("%s: requested_kernel needs entity_kind gpu_kernel and its kernels" % lab)
        for c in (t.get("manifest") or {}).get("cases") or []:
            calls = c.get("effective_calls")
            if (not c.get("sig") or isinstance(calls, bool) or not isinstance(calls, (int, float))
                    or calls < 0):
                problems.append("%s: every case needs a sig and a nonnegative effective_calls" % lab)
                break
    if not req.get("tasks") and not req.get("blocked_identities"):
        problems.append("the request names no task and no blocked identity")
    return req, root, problems


def scaffold(task, tdir, force):
    sc, man = task["scaffold"], task.get("manifest")
    if sc.get("family") in (None, "custom") and not sc.get("family_file"):
        return {"status": "skipped", "why": "custom family: write a contract (--family-file)"}
    if man is None:
        return {"status": "skipped", "why": "the request has no case manifest for this task"}
    if os.path.exists(os.path.join(tdir, "unittest.py")) and not force:
        return {"status": "kept", "why": "unittest.py exists; --force to regenerate"}
    mpath = os.path.join(tdir, "run", "ut_cases.manifest.json")
    rpath = os.path.join(tdir, "run", "phase_regimes.json")
    dump(man, mpath)
    dump(sc.get("phase_regimes") or {}, rpath)
    family_arg = (["--family-file", sc["family_file"]] if sc.get("family_file")
                  else ["--family", sc["family"]])
    cmd = [sys.executable, SCAFFOLD, "new", *family_arg, "--task", tdir,
           "--manifest", mpath, "--phase-regimes", rpath]
    if sc.get("target_callable"):
        cmd += ["--target", sc["target_callable"]]
    # scaffold_ut parses non-strings as JSON: a bool must arrive as `true`, not `True`
    for k, v in (sc.get("axes") or {}).items():
        cmd += ["--axis", "%s=%s" % (k, v if isinstance(v, str) else json.dumps(v))]
    for k, v in (sc.get("attrs") or {}).items():
        cmd += ["--attr", "%s=%s" % (k, v if isinstance(v, str) else json.dumps(v))]
    if force:
        cmd.append("--force")
    p = subprocess.run(cmd, capture_output=True, text=True)
    lines = [l for l in (p.stdout + p.stderr).splitlines() if l.strip()]
    return {"status": "ok" if p.returncode == 0 else "failed", "rc": p.returncode,
            "error": lines[-1] if p.returncode and lines else None,
            "command": cmd, "output": "\n".join(lines[-30:])}


def stamp_meta(tdir, task):
    """Write the request's server evidence and launch expectations into meta.json."""
    path = os.path.join(tdir, "meta.json")
    meta = load(path)
    if meta is None:
        return False
    if task.get("expected_launches") is not None:
        meta["expected_launches"] = task["expected_launches"]
    meta["server_preimports"] = task.get("server_preimports") or []
    ev = task.get("server_evidence") or {}
    if ev and "missing" not in ev:
        meta["server_evidence"] = ev
    if task.get("compile_variant_waivers") is not None:
        meta.setdefault("ut_contract", {})["compile_variant_waivers"] = task["compile_variant_waivers"]
    if task.get("provenance") is not None:
        meta["provenance"] = task["provenance"]
    if "invocation_counts" in load(os.path.join(tdir, "run", "ut_case_plan.json"), {}):
        files = meta.setdefault("kernel_extraction", {}).setdefault("source_delivery_files", [])
        if "run/ut_case_plan.json" not in files:
            files.append("run/ut_case_plan.json")
    dump(meta, path, indent=2)
    return True


def install(req, root, out, force=False):
    tasks = []
    for t in req.get("tasks") or []:
        tdir = os.path.join(out, "tasks", t["label"])
        run = os.path.join(tdir, "run")
        os.makedirs(run, exist_ok=True)
        src = os.path.join(root, "tasks", t["label"], "run")
        if os.path.isdir(src):
            shutil.rmtree(os.path.join(run, "server_calls"), ignore_errors=True)
            shutil.copytree(src, run, dirs_exist_ok=True)
        ev = t.get("server_evidence") or {}
        if ev and "missing" not in ev:
            shutil.copy2(COMPILE_WATCH, os.path.join(tdir, "compile_watch.py"))
        rk = dict(t["requested_kernel"], task_dir=tdir,
                  server_preimports=t.get("server_preimports") or [],
                  server_evidence=ev or {"missing": "the request carries no server evidence"})
        dump(rk, os.path.join(run, "requested_kernel.json"))
        sc = scaffold(t, tdir, force)
        stamp_meta(tdir, t)
        tasks.append({"label": t["label"], "task_dir": tdir,
                      "selected_identities": rk.get("selected_identities"),
                      "required_companions": [c.get("kernel") for c in rk.get("required_companions") or []],
                      "phases": rk.get("phases"), "ut_family": t["scaffold"].get("family"),
                      "target_callable": rk.get("target_callable"),
                      "planned_cases": t.get("planned_cases"),
                      "selection_pct_gpu_time": t.get("selection_pct_gpu_time"),
                      "requested_kernel": os.path.join(run, "requested_kernel.json"),
                      "server_evidence": rk["server_evidence"], "scaffold": sc})
    if os.path.isfile(os.path.join(root, "selection_targets.json")):
        shutil.copy2(os.path.join(root, "selection_targets.json"),
                     os.path.join(out, "selection_targets.json"))
    plan = {"contract": "agentx_task_plan/v1",
            "scope": "UT generation; installed by scaffold_tasks.py from ut_request.json",
            "request": os.path.join(root, "ut_request.json"), "tasks": tasks,
            "blocked_identities": req.get("blocked_identities") or [],
            "selected_overall_pct": req.get("selected_overall_pct"),
            "warnings": req.get("warnings") or []}
    dump(plan, os.path.join(out, "task_plan.json"))
    return plan


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--request", required=True)
    ap.add_argument("--out", default=os.environ.get("OUT"))
    ap.add_argument("--force", action="store_true", help="regenerate existing UT scaffolds")
    ap.add_argument("--check", action="store_true", help="verify the request, write nothing")
    a = ap.parse_args()
    req, root, problems = check_request(a.request)
    for p in problems:
        print("  ERROR: %s" % p)
    if problems:
        return 1
    if a.check:
        print("scaffold_tasks: request is consistent (%d tasks)" % len(req["tasks"]))
        return 0
    if not a.out:
        ap.error("--out or OUT is required")
    plan = install(req, root, os.path.abspath(a.out), a.force)
    for t in plan["tasks"]:
        sc = t["scaffold"]
        print("  %-44s %-18s scaffold %s%s" % (t["label"], t["ut_family"], sc["status"],
              ": " + (sc.get("error") or sc.get("why") or "") if sc["status"] != "ok" else ""))
    print("-> %s" % os.path.join(a.out, "task_plan.json"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
