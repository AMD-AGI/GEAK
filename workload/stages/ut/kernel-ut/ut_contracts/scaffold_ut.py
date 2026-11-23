#!/usr/bin/env python3
"""Create or refresh a task directory from a UT contract family.

    scaffold_ut.py list
    scaffold_ut.py new --family gemm_a8w8_blockscale --task $EVAL_DIR/tasks/<name> \
        --target pkg.module:entry --bind-module pkg.module --bind-file kernel_src/module.py \
        --attr fp8_dtype=float8_e4m3fn --axis n=7168 --axis k=2048 \
        --manifest cases.json [--phase-regimes regimes.json] [--override overrides.json]
    scaffold_ut.py sync-meta --task <task>
    scaffold_ut.py lint --task <task>
    scaffold_ut.py baseline-package --task <task> [--root <installed package dir>]

`new` copies the shared helpers (harness_lib, leg_runner, overlay_setup from $SHARED_TOOLS)
and this directory's ut_contract.py / reference_f64.py / ut_driver.py into the task, freezes
the family plus the task's attrs, const axes and justified overrides as run/ut_contract.json,
and writes meta.json, cases.py, unittest.py and run/ut_cases.json. It never overwrites an
existing file without --force. What it cannot know — the live call, the family's hooks, the
negative controls, the workload evidence — is left as FILL markers that fail loudly.

`baseline-package` records the installed package a whole-package bind (candidate_bind of a
package `__init__.py`) shadows, as run/baseline_package.json {package, root, files: {path:
sha256}}; run it where the package is installed (the serving image). The lint then requires every
file of kernel_src/<package>/ that differs from it to be an editable leaf or a frozen_differs
entry with its why.

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
# compile_watch.py records variants in the profiled server and replays them in the UT, so it
# lives with the stage-shared tools rather than in this skill
STAGE_SHARED = os.path.join(os.path.dirname(os.path.dirname(SKILL_ROOT)), "_shared")
SHARED_HELPERS = ("harness_lib.py", "leg_runner.py", "overlay_setup.py", "compile_watch.py",
                  "kernel_source_delivery.mjs", "rank_timing.py")
CONTRACT_HELPERS = ("ut_contract.py", "reference_f64.py", "ut_driver.py")
# families whose UT runs through the generic multi-rank harness (mono_harness.py, copied into the
# task as rank_harness.py) and a task-filled rank_model.py (templates/rank_model.py.tmpl)
MONO_FAMILIES = ("mono_layer",)
MONO_HARNESS = ("mono_harness.py", "rank_harness.py")


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


def rank_chain_timing(doc):
    """meta.timing for a family timed by shared/rank_timing.py, else None. `world` is the task's
    pinned world axis (the deployment's rank count: TP8 Muse Spark, TP2 DeepSeek V4.1 ...); a
    family may still state a default, but none is assumed -- a guessed rank count would time a
    topology the deployment does not run."""
    t = doc.get("timing") or {}
    if t.get("kind") != "rank_chain":
        return None
    pinned = ((doc.get("task") or {}).get("const_axes") or {}).get("world") or t.get("world")
    if not pinned:
        raise SystemExit("family %s is timed by rank_chain: pin the deployment's rank count "
                         "(--axis world=<tp>)" % doc.get("family"))
    return {"kind": "rank_chain", "world": int(pinned), "chain_length": int(t.get("chain_length", 8))}


def workload_cases(manifest):
    out = []
    for c in manifest.get("cases", []):
        if not c.get("timed"):
            continue
        out.append({"sig": c["sig"], "regime": c["regime"], "m": c.get("m"),
                    "effective_calls": c.get("effective_calls")})
    return out


def control_stubs(doc, bind_file):
    """FILL negative controls: one per required category (min_controls_per_category of them where
    the family asks for more), each carrying its category; a family without categories gets the
    single generic stub."""
    leaf = bind_file or "kernel_src/FILL.py"
    cats = U.control_categories(doc, (doc.get("task") or {}).get("attrs") or {})
    mins = doc.get("min_controls_per_category") or {}
    stub = {"leaf": leaf, "find": "FILL: one exact, unique source fragment",
            "replace": "FILL: its numerically wrong replacement",
            "expect": "FILL: the deviation this must produce; see the family's negative_controls"}
    if not cats:
        return [stub]
    out = []
    for c in cats:
        for k in range(int(mins.get(c, 1))):
            out.append(dict(stub, category=c, leaf="FILL: the editable leaf that holds the %s code" % c,
                            expect="FILL: the outputs a %s bug must break (#%d of %d)" % (c, k + 1, int(mins.get(c, 1)))))
    return out


def graph_phases(meta):
    """Phases that deploy under a CUDA graph, from meta.phase_regimes or meta.regime."""
    if meta.get("phase_regimes"):
        return sorted(p for p, r in meta["phase_regimes"].items() if (r or {}).get("cuda_graph", True))
    return ["DECODE"] if (meta.get("regime") or {}).get("cuda_graph", True) else []


def mono_sources(doc, meta, name, package):
    """(cases.py, rank_model.py) of a MONO_FAMILIES task."""
    oracles = doc.get("oracles") or {}
    defs, names = [], []
    for oname, o in oracles.items():
        defs.append('def oracle_%s(args):\n    """Oracle leg `%s` (contract oracles.%s): RankModel path %r on every rank."""\n'
                    '    return _step(args, %r)\n\n' % (oname, oname, oname, o["path"], o["path"]))
        names.append("%r: oracle_%s" % (oname, oname))
    with open(os.path.join(HERE, "templates", "mono_cases.py.tmpl")) as f:
        cases = f.read().format(task_name=name, family=doc["family"], target_callable=meta["target_callable"],
                                outputs=", ".join(doc["outputs"]),
                                oracle_defs="\n".join(defs) if defs else "# the contract declares no oracle legs\n",
                                oracle_map=", ".join(names), graph_phases=graph_phases(meta),
                                chain_length=meta["timing"]["chain_length"])
    ax = (doc.get("task") or {}).get("const_axes") or {}
    attrs = (doc.get("task") or {}).get("attrs") or {}

    def axis(n, default="None  # FILL"):
        return ax[n] if n in ax else default

    paths = ("deployed",) + tuple(dict.fromkeys(o["path"] for o in oracles.values()))
    if attrs.get("stream_layout") == "mhc":
        # mono_harness STREAM_SHAPE / STREAM_AUX: the [rows, hc, hidden] stream and its packed mixes
        hc = axis("hc", None)
        if hc is None:
            raise SystemExit("stream_layout mhc needs the const axis hc (--axis hc=<hc_mult>)")
        stream_hooks = ("STREAM_SHAPE = (%d, %s)        # mhc: [rows, hc, hidden] (contract stream_layout)\n"
                        "    STREAM_AUX = (%d, torch.float32)  # mhc: post (hc) | comb (hc x hc) | pre (hc) a row; "
                        "launch() takes and returns it\n" % (hc, axis("hidden"), hc * (hc + 2)))
    else:
        stream_hooks = "# STREAM_SHAPE / STREAM_AUX: flat stream [rows, HIDDEN] (contract stream_layout flat)\n"
    with open(os.path.join(HERE, "templates", "rank_model.py.tmpl")) as f:
        model = f.read().format(task_name=name, family=doc["family"], target_callable=meta["target_callable"],
                                world=meta["timing"]["world"], launches=axis("launches"), pools=axis("pools", 2),
                                hidden=axis("hidden"), kv_width=axis("kv_width"),
                                topk=(axis("topk") if attrs.get("layer_kind", "routed_moe") == "routed_moe" else None),
                                stream_dtype=attrs.get("stream_dtype", "float32"), paths=paths,
                                stream_hooks=stream_hooks,
                                package=package or "FILL_package")
    return cases, model


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
    doc.setdefault("input_generation", {"protocol": "weight_substreams/v1", "weight_cache_bytes": 2 << 30})
    task = os.path.abspath(a.task)
    name = os.path.basename(task)
    os.makedirs(os.path.join(task, "run"), exist_ok=True)
    os.makedirs(os.path.join(task, "kernel_src"), exist_ok=True)
    os.makedirs(os.path.join(task, "baseline_overlay"), exist_ok=True)

    for f in SHARED_HELPERS:
        src = os.path.join(SHARED_TOOLS, f)
        if not os.path.exists(src) and os.path.exists(os.path.join(STAGE_SHARED, f)):
            src = os.path.join(STAGE_SHARED, f)
        if not os.path.exists(src):
            raise SystemExit("shared helper %s not found under %s" % (f, SHARED_TOOLS))
        if a.force or not os.path.exists(os.path.join(task, f)):
            shutil.copy2(src, os.path.join(task, f))
            print("copy    %s" % os.path.join(task, f))
    for f in CONTRACT_HELPERS:
        if a.force or not os.path.exists(os.path.join(task, f)):
            shutil.copy2(os.path.join(HERE, f), os.path.join(task, f))
            print("copy    %s" % os.path.join(task, f))

    mono = doc["family"] in MONO_FAMILIES
    if mono and (a.force or not os.path.exists(os.path.join(task, MONO_HARNESS[1]))):
        shutil.copy2(os.path.join(HERE, MONO_HARNESS[0]), os.path.join(task, MONO_HARNESS[1]))
        print("copy    %s" % os.path.join(task, MONO_HARNESS[1]))

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
        "kernel_extraction": {"source_delivery_files": [*SHARED_HELPERS, *CONTRACT_HELPERS,
                                                        "run/ut_contract.json", "run/ut_cases.json"]},
        "random_draws": 3,
        "workload": {"kind": "agentx_trace_replay", "weighting": "observed",
                     "serving_weight_model": {"source": "observed_workload",
                                              "evidence_path": "FILL: path to observed_workload.json"},
                     "cases": workload_cases(manifest)},
        "ut_contract": {
            "family": doc["family"],
            "contract_sha256": U.contract_sha256(doc),
            "editable_leaves": [a.bind_file] if a.bind_file else [],
            "negative_controls": control_stubs(doc, a.bind_file),
            "reference_rows": doc.get("reference_rows", 256),
            "replay_tol": 0.02,
            "min_control_margin": 10.0,
        },
    }
    timing = rank_chain_timing(doc)
    if timing is not None:
        meta["timing"] = timing
    if phase_regimes is not None:
        meta["phase_regimes"] = phase_regimes
    else:
        # harness_lib reads cuda_graph (default true) / enforce_eager / compile. Graph-captured is
        # the safe default: it makes the replay gate mandatory until the launch flags say otherwise.
        meta["regime"] = {"cuda_graph": True, "compile": "eager",
                          "note": "FILL from the deployment launch flags (parse_regime.py), or use phase_regimes"}
    if mono:
        # the JIT cache directory keyed by the source each leg resolved (rank_harness.leg_env)
        meta["leg_env"] = "rank_harness:leg_env"
        meta["rank_harness"] = ("rank_harness.py (generic, never edited) and rank_model.py (the task's "
                                "RankModel): frozen task sources, outside the candidate overlay")
    _write(os.path.join(task, "meta.json"), json.dumps(meta, indent=2) + "\n", a.force)

    if mono:
        if timing is None:
            raise SystemExit("family %s must declare timing.kind rank_chain" % doc["family"])
        cases, model = mono_sources(doc, meta, name, a.bind_module)
        _write(os.path.join(task, "cases.py"), cases, a.force)
        _write(os.path.join(task, "rank_model.py"), model, a.force)
        with open(os.path.join(HERE, "templates", "unittest.py.tmpl")) as f:
            _write(os.path.join(task, "unittest.py"), f.read().format(task_name=name, family=doc["family"]), a.force)
        print("\nnext: fill rank_model.py (RankModel; worked example "
              "references/examples/muse_spark_target_rank_model.py), the call-site docstring of cases.py, "
              "run/ut_cases.json and meta negative_controls / evidence (references/mono-layer-ut.md), then "
              "`scaffold_ut.py sync-meta --task %s` and `python3 ut_contract.py %s`." % (task, task))
        return 0

    stubs, hook_map = _hook_stubs(doc)
    outputs = ", ".join(doc["outputs"])
    with open(os.path.join(HERE, "templates", "cases.py.tmpl")) as f:
        cases = f.read().format(task_name=name, family=doc["family"], target_callable=meta["target_callable"],
                                hook_stubs=stubs, hook_map=hook_map, outputs=outputs,
                                call_note=doc.get("call", ""))
    if timing is not None:
        with open(os.path.join(HERE, "templates", "rank_chain.py.tmpl")) as f:
            cases += f.read().format(world=timing["world"], chain_length=timing["chain_length"])
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


def cmd_baseline_package(a):
    task = os.path.abspath(a.task)
    with open(os.path.join(task, "meta.json")) as f:
        meta = json.load(f)
    if U.package_bind_dir(meta) is None:
        raise SystemExit("meta.candidate_bind is not a whole-package bind (a package __init__.py)")
    module = meta["candidate_bind"]["module"]
    root = a.root
    if not root:
        import importlib.util
        spec = importlib.util.find_spec(module)
        if spec is None or not spec.submodule_search_locations:
            raise SystemExit("%s is not an importable package here; pass --root" % module)
        root = list(spec.submodule_search_locations)[0]
        if os.path.abspath(root).startswith(task + os.sep):
            raise SystemExit("%s resolves inside the task (%s), not to the installed package"
                             % (module, root))
    files = U.tree_file_hashes(root)
    if not files:
        raise SystemExit("no files under %s" % root)
    rec = {"package": module, "root": os.path.abspath(root), "files": files}
    path = os.path.join(task, U.BASELINE_PACKAGE)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(rec, f, indent=1, sort_keys=True)
        f.write("\n")
    print("recorded %d files of %s (%s) in %s" % (len(files), module, root, path))
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
    p = sub.add_parser("baseline-package")
    p.add_argument("--task", required=True)
    p.add_argument("--root", help="the installed package directory (default: find_spec here)")
    p.set_defaults(fn=cmd_baseline_package)
    a = ap.parse_args(argv)
    return a.fn(a)


if __name__ == "__main__":
    sys.exit(main())
