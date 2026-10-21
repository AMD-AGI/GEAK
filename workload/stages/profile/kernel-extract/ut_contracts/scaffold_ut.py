#!/usr/bin/env python3
"""Create or refresh a task directory from a UT contract family.

    scaffold_ut.py list
    scaffold_ut.py new --family gemm_a8w8_blockscale --task $EVAL_DIR/tasks/<name> \
        --target pkg.module:entry --bind-module pkg.module --bind-file kernel_src/module.py \
        --attr fp8_dtype=float8_e4m3fn --axis n=7168 --axis k=2048 \
        --manifest cases.json [--phase-regimes regimes.json] [--override overrides.json]
    scaffold_ut.py sync-meta --task <task>
    scaffold_ut.py lint --task <task>

`new` copies the shared helpers (harness_lib, leg_runner, overlay_setup from $SHARED_TOOLS)
and this directory's ut_contract.py / reference_f64.py / ut_driver.py into the task, freezes
the family plus the task's attrs, const axes and justified overrides as run/ut_contract.json,
and writes meta.json, cases.py, unittest.py and run/ut_cases.json. It never overwrites an
existing file without --force. What it cannot know — the live call, the family's hooks, the
negative controls, the workload evidence — is left as FILL markers that fail loudly.

`sync-meta` re-derives meta.workload.cases from the manifest's timed cases and re-stamps the
contract hash; run it after editing run/ut_cases.json.
"""

import argparse
import json
import os
import shutil
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import ut_contract as U  # noqa: E402

SKILL_ROOT = os.path.dirname(HERE)
SHARED_TOOLS = os.environ.get("SHARED_TOOLS") or os.path.join(SKILL_ROOT, "shared")
SHARED_HELPERS = ("harness_lib.py", "leg_runner.py", "overlay_setup.py")
CONTRACT_HELPERS = ("ut_contract.py", "reference_f64.py", "ut_driver.py")


def _kv(items):
    out = {}
    for it in items or []:
        k, _, v = it.partition("=")
        if not k or not _:
            raise SystemExit("expected key=value, got %r" % it)
        out[k] = json.loads(v) if v[:1] in "[{\"0123456789-" or v in ("true", "false", "null") else v
    return out


def _write(path, text, force):
    if os.path.exists(path) and not force:
        print("keep    %s (exists; --force to replace)" % path)
        return False
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        f.write(text)
    print("write   %s" % path)
    return True


def _first_var_axis(doc):
    return next((n for n, a in doc["axes"].items() if a.get("var")), None)


def _normalize_manifest(doc, manifest):
    """Fill `m` from the first var axis and validate every case against the contract."""
    m_axis = _first_var_axis(doc)
    for c in manifest.get("cases", []):
        if c.get("m") is None and m_axis and m_axis in (c.get("axes") or {}):
            c["m"] = c["axes"][m_axis]
        U.symbols_for(doc, c.get("axes") or {})
    return manifest


def _stub_manifest(doc):
    """One example row at the smallest legal point: a placeholder the agent must replace."""
    axes = {n: a.get("lo", 1) for n, a in doc["axes"].items() if a.get("var")}
    return {"note": "FILL: replace with the observed cases (sig, regime, axes, effective_calls, "
                    "timed, random) from the workload evidence; replay_sigs per graph phase",
            "cases": [{"sig": "FILL_example", "regime": "DECODE", "axes": axes,
                       "effective_calls": 0, "timed": False, "random": True}],
            "replay_sigs": []}


def _hook_stubs(doc):
    stubs, names = [], []
    for name, what in (doc.get("hooks") or {}).items():
        names.append(name)
        layout = any((s.get("layout") or {}).get("hook") == name for s in doc["inputs"].values())
        sig = "(t, name, sym)" if layout else "(name, spec, shape, dtype, sym, gen, device)"
        stubs.append('def %s%s:\n    """%s"""\n    raise NotImplementedError("FILL hook %s")\n\n'
                     % (name, sig, what.replace('"""', "'''"), name))
    return "\n".join(stubs) if stubs else "# this family declares no hooks\n", \
        ", ".join("%r: %s" % (n, n) for n in names)


def workload_cases(manifest):
    return [{"sig": c["sig"], "regime": c["regime"], "m": c.get("m"),
             "effective_calls": c.get("effective_calls")}
            for c in manifest.get("cases", []) if c.get("timed")]


def cmd_list(a):
    for f in sorted(os.listdir(U.FAMILIES_DIR)):
        if f.endswith(".json"):
            d = U.load_family(f[:-5])
            print("%-24s %-11s %s" % (d["family"], d["op_kind"], d["description"][:110]))
    return 0


def cmd_new(a):
    family = U.load_family(a.family) if a.family else None
    if a.family_file:
        with open(a.family_file) as f:
            family = U.validate_family(json.load(f))
    if family is None:
        raise SystemExit("--family or --family-file is required")
    overrides = []
    if a.override:
        with open(a.override) as f:
            overrides = json.load(f)
    doc = U.freeze_task_contract(family, attrs=_kv(a.attr), const_axes=_kv(a.axis), overrides=overrides)
    task = os.path.abspath(a.task)
    name = os.path.basename(task)
    os.makedirs(os.path.join(task, "run"), exist_ok=True)
    os.makedirs(os.path.join(task, "kernel_src"), exist_ok=True)
    os.makedirs(os.path.join(task, "baseline_overlay"), exist_ok=True)

    for f in SHARED_HELPERS:
        src = os.path.join(SHARED_TOOLS, f)
        if not os.path.exists(src):
            raise SystemExit("shared helper %s not found under %s" % (f, SHARED_TOOLS))
        if a.force or not os.path.exists(os.path.join(task, f)):
            shutil.copy2(src, os.path.join(task, f))
            print("copy    %s" % os.path.join(task, f))
    for f in CONTRACT_HELPERS:
        if a.force or not os.path.exists(os.path.join(task, f)):
            shutil.copy2(os.path.join(HERE, f), os.path.join(task, f))
            print("copy    %s" % os.path.join(task, f))

    contract_path = os.path.join(task, "run", "ut_contract.json")
    _write(contract_path, json.dumps(doc, indent=2) + "\n", a.force)
    with open(contract_path) as f:
        doc = json.load(f)

    if a.manifest:
        with open(a.manifest) as f:
            manifest = _normalize_manifest(doc, json.load(f))
    else:
        manifest = _stub_manifest(doc)
    _write(os.path.join(task, "run", "ut_cases.json"), json.dumps(manifest, indent=2) + "\n", a.force)

    phase_regimes = None
    if a.phase_regimes:
        with open(a.phase_regimes) as f:
            phase_regimes = json.load(f)
    meta = {
        "task": name,
        "op_kind": doc["op_kind"],
        "target_callable": a.target or "FILL.module:entry",
        "candidate_bind": {"kind": "module", "module": a.bind_module or "FILL.module",
                           "file": a.bind_file or "kernel_src/FILL.py"},
        "random_draws": 3,
        "workload": {"kind": "agentx_trace_replay", "weighting": "observed",
                     "serving_weight_model": {"source": "observed_workload",
                                              "evidence_path": "FILL: path to observed_workload.json"},
                     "cases": workload_cases(manifest)},
        "ut_contract": {
            "family": doc["family"],
            "contract_sha256": U.contract_sha256(doc),
            "editable_leaves": [a.bind_file] if a.bind_file else [],
            "negative_controls": [{"leaf": a.bind_file or "kernel_src/FILL.py",
                                   "find": "FILL: one exact, unique source fragment",
                                   "replace": "FILL: its numerically wrong replacement",
                                   "expect": "FILL: the deviation this must produce; see the family's negative_controls"}],
            "reference_rows": doc.get("reference_rows", 256),
            "replay_tol": 0.02,
            "min_control_margin": 10.0,
        },
    }
    if phase_regimes is not None:
        meta["phase_regimes"] = phase_regimes
    else:
        # harness_lib reads cuda_graph (default true) / enforce_eager / compile. Graph-captured is
        # the safe default: it makes the replay gate mandatory until the launch flags say otherwise.
        meta["regime"] = {"cuda_graph": True, "compile": "eager",
                          "note": "FILL from the deployment launch flags (parse_regime.py), or use phase_regimes"}
    _write(os.path.join(task, "meta.json"), json.dumps(meta, indent=2) + "\n", a.force)

    stubs, hook_map = _hook_stubs(doc)
    outputs = ", ".join(doc["outputs"])
    with open(os.path.join(HERE, "templates", "cases.py.tmpl")) as f:
        cases = f.read().format(task_name=name, family=doc["family"], target_callable=meta["target_callable"],
                                hook_stubs=stubs, hook_map=hook_map, outputs=outputs,
                                call_note=doc.get("call", ""))
    _write(os.path.join(task, "cases.py"), cases, a.force)
    with open(os.path.join(HERE, "templates", "unittest.py.tmpl")) as f:
        ut = f.read().format(task_name=name, family=doc["family"])
    _write(os.path.join(task, "unittest.py"), ut, a.force)
    print("\nnext: fill the FILL markers (cases.py hooks + call, run/ut_cases.json, meta "
          "negative_controls/evidence), then `scaffold_ut.py sync-meta --task %s` and "
          "`python3 ut_contract.py %s`." % (task, task))
    return 0


def cmd_sync_meta(a):
    task = os.path.abspath(a.task)
    doc = U.load_task_contract(task)
    with open(os.path.join(task, "run", "ut_cases.json")) as f:
        manifest = _normalize_manifest(doc, json.load(f))
    with open(os.path.join(task, "run", "ut_cases.json"), "w") as f:
        json.dump(manifest, f, indent=2)
        f.write("\n")
    path = os.path.join(task, "meta.json")
    with open(path) as f:
        meta = json.load(f)
    meta.setdefault("workload", {})["cases"] = workload_cases(manifest)
    meta.setdefault("ut_contract", {})["contract_sha256"] = U.contract_sha256(doc)
    meta["ut_contract"]["family"] = doc["family"]
    with open(path, "w") as f:
        json.dump(meta, f, indent=2)
        f.write("\n")
    print("synced %d timed cases into %s" % (len(meta["workload"]["cases"]), path))
    return 0


def cmd_lint(a):
    return U.main([a.task])


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("list").set_defaults(fn=cmd_list)
    p = sub.add_parser("new")
    p.add_argument("--family")
    p.add_argument("--family-file", help="a task-authored contract for an op no family covers")
    p.add_argument("--task", required=True)
    p.add_argument("--target")
    p.add_argument("--bind-module")
    p.add_argument("--bind-file")
    p.add_argument("--attr", action="append")
    p.add_argument("--axis", action="append", help="const axis pinned for the task, e.g. n=7168")
    p.add_argument("--manifest")
    p.add_argument("--phase-regimes")
    p.add_argument("--override", help="JSON list of {path, value, why}")
    p.add_argument("--force", action="store_true")
    p.set_defaults(fn=cmd_new)
    p = sub.add_parser("sync-meta")
    p.add_argument("--task", required=True)
    p.set_defaults(fn=cmd_sync_meta)
    p = sub.add_parser("lint")
    p.add_argument("--task", required=True)
    p.set_defaults(fn=cmd_lint)
    a = ap.parse_args(argv)
    return a.fn(a)


if __name__ == "__main__":
    sys.exit(main())
