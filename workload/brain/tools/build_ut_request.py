#!/usr/bin/env python3
"""Reviewed task_decls.json + one or more profile deliveries -> ut_request.json, the only
input of the UT stage.

    python3 build_ut_request.py --kernels <profile out>/kernels.json \
        --decls task_decls.json --out <request dir> [--check] [--accept-draft]

Task boundaries are decided here, by whoever reviews task_decls.json (draft_task_decls.py
writes the first version); the UT stage only writes and qualifies what the request names.
Everything else a task needs is evidence the profile already measured, assembled the same
way every run, so sibling tasks cannot disagree about a census list or a denominator:

  <out>/ut_request.json      tasks[]: the delivery-contract KERNEL object, the family and
                             scaffold arguments, the concrete case manifest (axes and
                             effective_calls per case), replay sigs, server preimports,
                             compile-variant waivers and expected launches; blocked
                             identities with their reasons; and the SHA-256 of every file
                             below
  <out>/tasks/<label>/run/   ut_case_plan.json (plan_ut_cases.py), server_variants.json and
                             server_calls.json (+ server_calls/*.pt): the Triton variants
                             the probe server built for the task's kernels and the entry
                             calls that built them (evidence/compile_variants.json, from
                             process start: warmup and graph capture included)
  <out>/selection_targets.json   the profile's selection, carried for downstream stages

server_preimports (decls `server_preimports`, default by framework: sglang ->
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
import hashlib
import json
import os
import re
import shutil
import sys
from collections import Counter

HERE = os.path.dirname(os.path.abspath(__file__))
STAGES = os.path.join(os.path.dirname(os.path.dirname(HERE)), "stages")
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(STAGES, "_shared"))
sys.path.insert(0, os.path.join(STAGES, "profile"))
import compile_watch  # noqa: E402
from plan_ut_cases import invocation_basis, plan as plan_cases  # noqa: E402

# the UT stage's catalogue of contract families a task may name
FAMILIES = os.path.join(STAGES, "ut", "kernel-ut", "ut_contracts", "families")
CONTRACT = "workload/ut_request/v1"
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
        try:
            invocation_basis(t)
        except ValueError as exc:
            errs.append(str(exc))
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
    def base_sig(c):
        return "%s_%s%s" % (c["phase"], c["shape"], "" if c["kind"] == "point" else "_" + c["kind"])
    counts = Counter(base_sig(c) for c in cases)
    used = set()
    for c in cases:
        expr = (task.get("case_axes") or {}).get(c["phase"])
        if not expr:
            return None
        names = {"shape": c["shape"], "context": c.get("context") or c["shape"]}
        axes = {ax: int(round(eval(e, {"__builtins__": {}}, names)))  # noqa: S307
                for ax, e in expr.items()}
        sig = base_sig(c)
        if counts[sig] > 1:
            # A shape can have several measured contexts.
            # Preserve each point's independent weight and evidence identity.
            sig += "__" + c["case_id"]
        if sig in used:
            raise ValueError("duplicate case identity: %s" % sig)
        used.add(sig)
        row = {"sig": sig, "regime": c["phase"], "axes": axes,
               "effective_calls": c["effective_calls"], "timed": c["kind"] != "envelope",
               "random": True, "kind": c["kind"],
               "evidence": c.get("shape_manifest") or "ut_case_plan.json#" + c["case_id"]}
        for key in ("invocation_count_basis", "invocations_per_instance", "kernel_launches_per_instance"):
            if key in c:
                row[key] = c[key]
        out.append(row)
        by_phase.setdefault(c["phase"], []).append((c["shape"], sig))
    replay = []
    for ph, rows in by_phase.items():
        if (task.get("phase_regimes") or {}).get(ph, {}).get("cuda_graph"):
            rows.sort()
            replay += sorted({rows[0][1], rows[-1][1]})
    return {"cases": out, "replay_sigs": replay}


def seam_depth(t, source_map):
    """Depth of the task's target_callable in the observed launch stacks, counted from the
    outermost frame (deeper = larger, the order delivery_contract.mjs ranks candidates by).
    source_map.json lists each kernel's frames innermost first. None when no stack of the
    task's kernels contains the callable."""
    target, found = t.get("target_callable"), []
    kernels = (source_map or {}).get("kernels") or {}
    for k in t.get("kernels") or []:
        frames = (kernels.get(k) or {}).get("frames") or []
        for i, f in enumerate(frames):
            if f.get("target_callable") == target:
                found.append(len(frames) - i)
                break
    return max(found) if found else None


def seam(t, device_kernels, depth):
    """One delivery-contract seam candidate for the task's whole-op entry: the gate keeps a
    candidate only with a finite depth and device_kernels naming the kernel it must launch
    (delivery_contract.mjs selectionCandidatesForHead)."""
    if not t.get("target_callable"):
        return []
    return [{"target_callable": t.get("target_callable"), "path": t.get("target_file"),
             "sha256": t.get("target_sha256"), "role": "op_seam",
             "depth": depth if depth is not None else 1,
             "depth_basis": ("evidence/source_map.json frame position" if depth is not None
                             else "single candidate; callable not in a recorded stack"),
             "device_kernels": list(dict.fromkeys(device_kernels)),
             "evidence": "innermost serving-stack frame common to the task's kernels "
                         "(evidence/source_map.json)"}]


def joint_ut(t, rank, depth=None):
    """task_decls joint_phases -> delivery-contract joint_ut.phases[] rows (anchor = the
    row's heaviest kernel; a row of one kernel is not a chain and is reported apart)."""
    rows, single = [], []
    js = t.get("joint_phases") or []
    branches = {k for j in js for k in j.get("required_kernels") or []}
    if js and (t.get("selection_scope") == "dispatch" or
               all(len(j.get("required_kernels") or []) == 1 for j in js) and len(branches) > 1):
        # Dispatch is still one complete operator; a branch may itself be a chain.
        return None, [], {"phases": [
            {"regime": j["regime"],
             "device_kernel": max(j["required_kernels"], key=lambda k: rank.get(k, {}).get("mass_us", 0)),
             **({"required_device_kernels": j["required_kernels"]}
                if len(j["required_kernels"]) > 1 else {}),
             "served_phase": j.get("phase"), "owned_point_ids": j.get("points"),
             "seam_candidates": seam(t, j["required_kernels"], depth)} for j in js]}
    for j in js:
        ks = list(dict.fromkeys(j.get("required_kernels") or []))
        if len(ks) < 2:
            single.append(j)
            continue
        rows.append({"regime": j["regime"],
                     "device_kernel": max(ks, key=lambda k: rank.get(k, {}).get("mass_us", 0)),
                     "required_device_kernels": ks, "served_phase": j.get("phase"),
                     "owned_point_ids": j.get("points"), "seam_candidates": seam(t, ks, depth)})
    return ({"phases": rows} if rows else None), single, None


def frameworks(task, cv):
    """Serving frameworks on the task's launch paths (its entry or any server stack)."""
    mods = [(task.get("target_callable") or "").split(".")[0]]
    for r in (cv or {}).get("variants", []):
        mods += [f.split(".")[0] for f in r.get("stack") or []]
    return sorted({m for m in mods if m in SERVER_PREIMPORTS})


# Families whose entry is a monokernel runner that falls back to the framework's own module
# for some layers (Muse Spark: MuseMonoRunner.launch runs MuseSpark1xDecoderLayer.forward for
# the bf16-expert layers 0 and 61). The Triton variants that fallback compiles sit under the
# entry on the stack, so they are sliced into the task, but no monokernel launch builds them.
FALLBACK_PATH_FAMILIES = {"mono_layer"}


def _top(mod):
    return mod.split(".")[0]


def fallback_path_why(rec, kernel_names, target_callable):
    """Why a server variant is the entry's fallback path, not the monokernel's; None if it is not.

    All three must hold: (1) the variant is no kernel of the task; (2) the frame the entry
    called (one frame inside it on the recorded stack, innermost first) is a `forward` in a
    package other than the entry's -- the entry handed the layer to the framework module;
    (3) it was compiled eagerly and not inside graph capture, so nothing captured the
    monokernel launches depend on it."""
    if rec.get("name") in kernel_names or not target_callable:
        return None
    if not rec.get("eager") or rec.get("in_graph_capture"):
        return None
    mod, _, qual = target_callable.partition(":")
    leaf = qual.rsplit(".", 1)[-1]
    stack = rec.get("stack") or []
    at = None
    for i, fr in enumerate(stack):     # the entry's frame, matched as compile_watch.belongs does
        fmod, _, fqual = fr.partition(":")
        if fmod == mod and (fqual == qual or fqual.rsplit(".", 1)[-1] == leaf):
            at = i
            break
    if not at:              # entry absent, or the entry compiled it itself (index 0)
        return None
    callee = stack[at - 1]
    cmod, _, cqual = callee.partition(":")
    if _top(cmod) == _top(mod) or cqual.rsplit(".", 1)[-1] != "forward":
        return None
    return ("fallback path of %s, not its monokernel: %s compiled under %s, which the entry "
            "calls in place of the monokernel (a module forward outside the %s package), "
            "eager=True and in_graph_capture=False on the server -- no monokernel launch "
            "builds it, so the task's UT cannot; waived by build_ut_request (family fallback "
            "rule), the variant stays in run/server_variants.json"
            % (target_callable, rec.get("name"), callee, _top(mod)))


def merge_waivers(declared, auto):
    """task_decls compile_variant_waivers first (theirs win on an id), then the automatic
    ones. None when neither has any, so stamp_meta leaves an existing meta alone."""
    if declared is None and not auto:
        return None
    out = list(declared or [])
    have = {w.get("id") for w in out}
    out += [w for w in auto or [] if w["id"] not in have]
    return out


# a recorded call larger than this is referenced where it is, never copied into the task
SERVER_CALL_COPY_MAX = int(os.environ.get("SERVER_CALL_COPY_MAX", str(256 << 20)))


def server_evidence(eval_dir, task, tdir):
    """The task's slice of evidence/compile_variants.json -> <tdir>/run/server_variants.json
    and run/server_calls.json (+ run/server_calls/<variant>.pt), the files the UT stage
    drops into the task's run/. Returns the meta block."""
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
    slim = [{k: v for k, v in r.items() if k not in ("frames", "tensors", "tensors_b64")} for r in recs]
    fallback = []
    if task.get("ut_family") in FALLBACK_PATH_FAMILIES:
        for r in recs:
            why = fallback_path_why(r, names, target)
            if why:
                fallback.append({"id": r["id"], "why": why})
    run = os.path.join(tdir, "run")
    os.makedirs(run, exist_ok=True)
    dump({"contract": "kernel_extract/server_variants/v1",
               "source": os.path.join(eval_dir, "evidence", "compile_variants.json"),
               "server_mode": cv.get("server_mode"), "env": cv.get("env") or {},
               "kernels": sorted(names), "target_callable": target, "variants": slim,
               **({"fallback_path_variants": [w["id"] for w in fallback]} if fallback else {})},
         os.path.join(run, "server_variants.json"))
    calls, unmatched, seen, referenced = [], [], set(), []
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
        if r.get("tensors_b64"):
            pass        # travels inline in server_calls.json: nothing to copy, no .pt in the task
        elif r.get("tensors"):
            src = os.path.join(cv["watch_dir"], r["tensors"])
            if os.path.exists(src) and os.path.getsize(src) > SERVER_CALL_COPY_MAX:
                # Evidence recorded before bulk payloads were sampled can hold whole weight
                # tensors (0.86 GB a call, 185 GB for one task). A task directory is copied by
                # every retain/materialize, so a large call stays in the read-only store and
                # is referenced by absolute path (leg_runner joins it onto the task dir).
                rel = os.path.abspath(src)
                referenced.append(os.path.getsize(src))
            elif os.path.exists(src):
                os.makedirs(os.path.join(run, "server_calls"), exist_ok=True)
                rel = os.path.join("run", "server_calls", r["id"] + ".pt")
                shutil.copy2(src, os.path.join(tdir, rel))
        calls.append({"_key": key, "id": r["id"], "variants": [r["id"]], "kernel": r["name"],
                      "callable": fr["callable"], "file": fr.get("file"), "line": fr.get("line"),
                      "positional": fr["positional"], "args": fr["args"], "tensors": rel,
                      **({"tensors_b64": r["tensors_b64"]} if r.get("tensors_b64") else {}),
                      "in_graph_capture": bool(r.get("in_graph_capture") and not r.get("eager")),
                      "before_ready": r.get("before_ready")})
    for c in calls:
        c.pop("_key")
    dump({"contract": "kernel_extract/server_calls/v1", "target_callable": target,
          "calls": calls, "unmatched_variants": unmatched}, os.path.join(run, "server_calls.json"))
    return {"server_preimports": pre, "fallback_waivers": fallback,
            "server_evidence": {"compile_variants": "run/server_variants.json",
                                "server_calls": "run/server_calls.json",
                                "server_mode": cv.get("server_mode"),
                                "variants": len(slim), "calls": len(calls),
                                "fallback_path_variants_waived": len(fallback),
                                "variants_without_entry_frame": len(unmatched),
                                "calls_referenced_in_place": len(referenced),
                                "bytes_referenced_in_place": sum(referenced)}}


def expected_launches(task, launched, phase_of):
    """{phase: kernels} -- every task kernel the server launched at that phase's points.
    ut_driver's `launches` gate requires the UT's baseline to launch each of them."""
    names = set(task.get("kernels") or []) | {c.get("kernel") for c in task.get("companions") or []}
    out = {}
    for pid, ks in launched.items():
        ph = phase_of.get(pid.split("@")[0])
        if ph:
            out.setdefault(ph, set()).update(ks & names)
    return {ph: sorted(ks) for ph, ks in sorted(out.items()) if ks}


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def build(eval_dir, decls, out, accept_draft=False, check_only=False):
    ev = lambda *p: os.path.join(eval_dir, *p)  # noqa: E731
    sel = load(ev("selection_targets.json"), {})
    targets = [t["id"] for t in sel.get("targets", [])]
    shapes_dir = ev("evidence", "shapes")
    launched = launched_at(shapes_dir)
    errs, warns = lint(decls, targets, launched, accept_draft)
    points = load(ev("sampling_points_final.json"))
    if not points:
        errs.append("sampling_points_final.json is missing (relabel_points.py)")
    cases = {}
    if not errs:
        try:
            cases = plan_cases(points, shapes_dir, decls["tasks"])["tasks"]
        except (KeyError, ValueError, TypeError) as exc:
            errs.append(str(exc))
    manifests = {}
    for t in decls.get("tasks", []) if not errs else []:
        try:
            manifests[t["label"]] = manifest(t, cases.get(t["label"], {"cases": []})["cases"])
        except (ValueError, NameError, SyntaxError, TypeError) as exc:
            errs.append("%s: case_axes cannot expand: %s" % (t["label"], exc))
        if manifests.get(t["label"]) is None and not errs:
            warns.append("%s: case_axes not set for every owned phase; the UT stage can only "
                         "write a custom harness for it" % t["label"])
    if errs or check_only:
        return {"errors": errs, "warnings": warns}

    phase_of = {p["point_id"]: p["phase"] for p in points.get("points", [])}
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
    source_map = load(ev("evidence", "source_map.json"), {})
    shutil.rmtree(os.path.join(out, "tasks"), ignore_errors=True)
    tasks = []
    for t in decls["tasks"]:
        tdir = os.path.join(out, "tasks", t["label"])
        os.makedirs(os.path.join(tdir, "run"), exist_ok=True)
        tc = cases.get(t["label"], {"cases": []})
        owned = sorted({c["point_id"] for c in tc["cases"] if c.get("point_id")})
        depth = seam_depth(t, source_map)
        joint, single, dispatch = joint_ut(t, rank, depth)
        companions = [c["kernel"] for c in t.get("companions") or [] if c.get("kernel")]
        # The KERNEL object of references/delivery-contract.md: the delivery gate
        # (kernel_extraction_delivery.mjs --kernel) refuses anything without entity_kind
        # gpu_kernel, and keeps only seam candidates with a depth and the kernels they launch.
        # The UT stage adds task_dir and the server evidence it installs.
        req = {
            "contract": "kernel_extract/requested_kernel/v1",
            "entity_kind": "gpu_kernel", "short_name": t["label"], "label": t["label"],
            "selected_identities": t["kernels"], "selected_device_kernels": t["kernels"],
            "device_kernel": max(t["kernels"], key=lambda k: rank.get(k, {}).get("mass_us", 0)),
            "required_companions": t.get("companions") or [],
            "invocation_anchors": t.get("invocation_anchors"),
            "invocation_count_basis": tc.get("invocation_count_basis"),
            "phases": t.get("phases"), "owned_point_ids": owned,
            "ut_family": t["ut_family"], "ut_family_why": t.get("ut_family_why"),
            "target_callable": t.get("target_callable"), "target_file": t.get("target_file"),
            "target_sha256": t.get("target_sha256"), "launching_ops": t.get("launching_ops"),
            "seam_candidates": seam(t, t["kernels"] + companions, depth),
            "joint_ut": joint,
            "single_kernel_regimes": single or None,
            "dispatch_ut": dispatch,
            "phase_regimes": t.get("phase_regimes"),
            "selection_pct_gpu_time": round(sum(rank.get(k, {}).get("selection_pct_gpu_time", 0)
                                                for k in t["kernels"]), 4),
            "source_delivery_required": True,
            "required_gpu_count": int(t.get("required_gpu_count") or 1),
            "capture_required": False,
            "shared": shared,
            "rules": ["hold one card; start no server, take no capture -- if evidence is "
                      "missing, return blocked naming it",
                      "write only inside task_dir"],
        }
        dump(tc, os.path.join(tdir, "run", "ut_case_plan.json"))
        se = server_evidence(eval_dir, t, tdir)
        kinds = {}
        for c in tc["cases"]:
            kinds[c["kind"]] = kinds.get(c["kind"], 0) + 1
        tasks.append({
            "label": t["label"], "requested_kernel": req,
            "scaffold": {"family": t["ut_family"], "family_file": t.get("family_file"),
                         "target_callable": t.get("target_callable"),
                         "axes": t.get("axes") or {}, "attrs": t.get("attrs") or {},
                         "phase_regimes": t.get("phase_regimes") or {}},
            "manifest": manifests.get(t["label"]),
            "planned_cases": dict(kinds, total=len(tc["cases"])),
            "proxy_mass_outside_every_case_pct": tc.get("proxy_mass_outside_every_case_pct"),
            "server_preimports": se["server_preimports"],
            "server_evidence": se.get("server_evidence") or {"missing": se.get("why")},
            "compile_variant_waivers": merge_waivers(t.get("compile_variant_waivers"),
                                                     se.get("fallback_waivers")),
            "expected_launches": expected_launches(t, launched, phase_of),
            "required_gpu_count": req["required_gpu_count"],
            "selection_pct_gpu_time": req["selection_pct_gpu_time"]})
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
    if os.path.exists(ev("selection_targets.json")):
        shutil.copy2(ev("selection_targets.json"), os.path.join(out, "selection_targets.json"))
    files = {}
    for root, _dirs, names in os.walk(out):
        for n in names:
            full = os.path.join(root, n)
            rel = os.path.relpath(full, out)
            if rel != "ut_request.json":
                files[rel] = {"sha256": sha256(full), "bytes": os.path.getsize(full)}
    request = {"contract": CONTRACT, "decls": decls.get("_path"),
               "profile": {"eval_dir": os.path.abspath(eval_dir)},
               "tasks": tasks, "blocked_identities": blocked,
               "selected_overall_pct": round(sum(t.get("overall_pct") or 0
                                                 for t in sel.get("targets", [])), 4),
               "warnings": warns, "files": dict(sorted(files.items()))}
    dump(request, os.path.join(out, "ut_request.json"))
    return {"errors": [], "warnings": warns, "request": request}


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--kernels", required=True, help="the profile's kernels.json")
    ap.add_argument("--decls", required=True, help="reviewed task_decls.json")
    ap.add_argument("--out", required=True, help="request directory (ut_request.json + files)")
    ap.add_argument("--check", action="store_true", help="lint only, write nothing")
    ap.add_argument("--accept-draft", action="store_true",
                    help="ignore needs_review (tests and dry runs only)")
    ap.add_argument("--no-verify", action="store_true",
                    help="skip re-deriving kernels.json from the profile evidence")
    a = ap.parse_args()
    if a.no_verify:
        eval_dir = load(a.kernels, {}).get("profile_root")
    else:
        from kernels import verify
        try:
            eval_dir = verify(a.kernels)["profile_root"]
        except (ValueError, OSError, KeyError, TypeError) as exc:
            raise SystemExit("build_ut_request: %s" % exc)
    decls = load(a.decls)
    if not decls:
        raise SystemExit("build_ut_request: no declaration at %s (draft_task_decls.py)" % a.decls)
    decls["_path"] = os.path.abspath(a.decls)
    os.makedirs(a.out, exist_ok=True)
    res = build(eval_dir, decls, os.path.abspath(a.out), a.accept_draft, a.check)
    for w in res["warnings"]:
        print("  warning: %s" % w)
    if res["errors"]:
        for e in res["errors"]:
            print("  ERROR: %s" % e)
        print("build_ut_request: %d error(s); nothing written" % len(res["errors"]))
        return 1
    if a.check:
        print("build_ut_request: declaration is consistent")
        return 0
    for t in res["request"]["tasks"]:
        print("  %-44s %-18s cases %s" % (t["label"], t["scaffold"]["family"],
                                          json.dumps(t["planned_cases"])))
    print("-> %s" % os.path.join(a.out, "ut_request.json"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
