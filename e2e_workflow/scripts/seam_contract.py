#!/usr/bin/env python3
"""seam_contract.py -- machine-checkable contracts for the two things an extraction ASSERTS but the
orchestrator has never been able to CHECK: (1) what the speedup denominator actually is, and (2) what
call contract an authored kernel has to satisfy to be rebindable at the live seam.

Why this file exists
--------------------
The role prompts already carry these rules as prose (kernel_extractor.md: "THE BASELINE LEG IS ALWAYS
THE FROZEN REAL ONLINE KERNEL", "never fabricate an oracle", "prove engagement before authoring").
Prose rules are followed on some runs and not on others, and the orchestrator's only check was
`typeof baseline_callable === 'string' && !== ''` -- which a task-local `baseline_src.attn_ref:
attention_forward` scaffold satisfies perfectly. A run can then post a large isolated speedup measured
against a pure-torch strawman it wrote itself, spend the whole kernel budget on it, and only discover
at integrate time that the number was never going to carry end-to-end.

Both problems are op-kind-agnostic and both are decidable by reflection, so they belong in code:

  INV-1 denominator identity  -- the object named by meta.baseline_callable must be importable from
                                 OUTSIDE the task dir, live in an installed distribution, and be the
                                 same seam as (or an observed callee of) meta.target_callable.
  INV-2 binding contract      -- the live callable's signature is captured mechanically and becomes
                                 THE contract. An authored entry is checked against it BEFORE any
                                 authoring budget is spent, so "the overlay cannot bind" is caught in
                                 seconds by inspect.signature instead of in hours by a serving A/B.

Nothing here knows about attention, MoE, GEMM or any backend. It only knows `module:attr`.

Usage (the extractor runs this and pastes the JSON into its return value):
    python3 seam_contract.py --task-dir <dir> --mode both --json
    python3 seam_contract.py --spec pkg.mod:fn --mode binding --json
    python3 seam_contract.py --task-dir <dir> --mode entry --out entry_contract.py

Exit code is 0 when every requested contract holds, 1 otherwise -- so a shell caller fails closed too.
Stdlib only (importing the seam itself may of course pull in torch; that is the caller's environment).
"""
from __future__ import annotations

import argparse
import importlib
import inspect
import json
import os
import site
import sys
import sysconfig

CONTRACT_VERSION = 1

# Parameter names that conventionally denote a caller-provided output buffer written IN PLACE. Used
# only to RAISE a flag (out_params.evidence == "name_convention"); runtime evidence from the capture
# step, when present in meta, overrides it. Kept deliberately small and generic.
_OUT_PARAM_NAMES = {"out", "output", "o", "dst", "dest", "y", "result", "out_tensor", "output_tensor"}


# --------------------------------------------------------------------------------- spec resolution
def parse_spec(spec):
    """'pkg.mod:attr' or 'pkg.mod.attr' -> (module_name, attr_path). Returns (None, None) if empty."""
    if not spec or not str(spec).strip():
        return None, None
    s = str(spec).strip()
    if ":" in s:
        mod, _, attr = s.partition(":")
        return mod.strip(), attr.strip()
    # dotted form: last component is the attribute
    if "." not in s:
        return None, None
    mod, _, attr = s.rpartition(".")
    return mod.strip(), attr.strip()


def resolve_spec(spec):
    """Import and resolve a module:attr spec. Never raises -- returns a verdict dict."""
    mod_name, attr_path = parse_spec(spec)
    if not mod_name or not attr_path:
        return {"ok": False, "spec": spec, "error": "unparseable spec (want 'module:attr')"}
    try:
        # The extractor freezes files and validates in the same process; without this, a module
        # written moments ago is invisible to the import system's cached directory listings.
        importlib.invalidate_caches()
        mod = importlib.import_module(mod_name)
    except Exception as e:  # noqa: BLE001 -- any import failure is a resolution failure
        return {"ok": False, "spec": spec, "module": mod_name, "error": f"import failed: {e!r}"}
    obj = mod
    for part in attr_path.split("."):
        if not hasattr(obj, part):
            return {"ok": False, "spec": spec, "module": mod_name,
                    "error": f"module has no attribute {attr_path!r}"}
        obj = getattr(obj, part)
    try:
        file = inspect.getfile(obj)
    except Exception:  # builtins / C extensions have no source file
        file = getattr(sys.modules.get(getattr(obj, "__module__", ""), None), "__file__", None)
    return {"ok": True, "spec": spec, "module": mod_name, "attr": attr_path, "obj": obj,
            "file": os.path.realpath(file) if file else None,
            "qualname": getattr(obj, "__qualname__", getattr(obj, "__name__", str(obj))),
            "callable": callable(obj)}


# --------------------------------------------------------------------------------- origin analysis
def _site_dirs():
    dirs = []
    for fn in ("purelib", "platlib"):
        try:
            p = sysconfig.get_paths().get(fn)
            if p:
                dirs.append(os.path.realpath(p))
        except Exception:
            pass
    try:
        dirs.extend(os.path.realpath(p) for p in site.getsitepackages())
    except Exception:
        pass
    try:
        usp = site.getusersitepackages()
        if isinstance(usp, str):
            dirs.append(os.path.realpath(usp))
    except Exception:
        pass
    # Overlay / editable / vendored install roots that sysconfig does not know about. The overlay the
    # integrator builds is a legitimate install location, so a run must be able to declare it rather
    # than have every baseline in it classified `unknown` and fail-closed for the wrong reason.
    for extra in (os.environ.get("GEAK_SEAM_SITE_DIRS", "") or "").split(os.pathsep):
        if extra.strip():
            dirs.append(os.path.realpath(extra.strip()))
    return sorted(set(d for d in dirs if d))


def _stdlib_dirs():
    out = []
    for fn in ("stdlib", "platstdlib"):
        try:
            p = sysconfig.get_paths().get(fn)
            if p:
                out.append(os.path.realpath(p))
        except Exception:
            pass
    return sorted(set(out))


def _under(path, root):
    if not path or not root:
        return False
    path = os.path.realpath(path)
    root = os.path.realpath(root)
    return path == root or path.startswith(root.rstrip(os.sep) + os.sep)


def origin_of(path, task_dir=None, eval_dir=None):
    """Classify where a resolved callable's source file lives.

    Order matters: a file under the task dir is task_local EVEN IF the task dir happens to sit inside
    site-packages, because the point of the check is "did the extraction time itself against something
    it wrote". `unknown` is a FAILING classification, not a benign one -- an anonymous path is exactly
    what a scaffold dropped in cwd looks like.
    """
    if not path:
        return {"kind": "unresolved", "distribution": None, "path": None}
    rp = os.path.realpath(path)
    if task_dir and _under(rp, task_dir):
        return {"kind": "task_local", "distribution": None, "path": rp}
    if eval_dir and _under(rp, eval_dir):
        return {"kind": "eval_local", "distribution": None, "path": rp}
    for d in _stdlib_dirs():
        if _under(rp, d) and not any(_under(rp, s) for s in _site_dirs()):
            return {"kind": "stdlib", "distribution": None, "path": rp}
    for d in _site_dirs():
        if _under(rp, d):
            return {"kind": "installed", "distribution": _distribution_for(rp, d), "path": rp}
    return {"kind": "unknown", "distribution": None, "path": rp}


def _distribution_for(realpath, site_dir):
    """Best-effort distribution name: the first path component under site-packages."""
    rel = os.path.relpath(realpath, site_dir)
    top = rel.split(os.sep)[0]
    top = top[:-3] if top.endswith(".py") else top
    try:
        import importlib.metadata as md
        for dist, tops in (md.packages_distributions() or {}).items():
            if dist == top and tops:
                return tops[0]
    except Exception:
        pass
    return top or None


# ------------------------------------------------------------------------- INV-1 baseline identity
def validate_baseline(task_dir=None, meta=None, eval_dir=None):
    """INV-1: is meta.baseline_callable a legitimate speedup DENOMINATOR?

    Six checks, each independently reportable so a failure says which one broke. A `False` on any of
    them means the isolated speedup this task will produce is not comparable to anything the live
    server runs, and the task must be dropped (`editable:false`) rather than authored against.
    """
    meta = dict(meta or {})
    checks = []

    def add(cid, ok, detail):
        checks.append({"id": cid, "ok": bool(ok), "detail": detail})

    base_spec = (meta.get("baseline_callable") or "").strip()
    tgt_spec = (meta.get("target_callable") or "").strip()

    add("B1_baseline_declared", bool(base_spec),
        f"meta.baseline_callable={base_spec!r}" if base_spec else "meta.baseline_callable is missing/empty")

    add("B6_not_synthesized", meta.get("synthesized") is not True,
        "meta.synthesized is true -- the oracle was fabricated, it cannot be the denominator"
        if meta.get("synthesized") is True else "meta.synthesized is not true")

    base_res = resolve_spec(base_spec) if base_spec else {"ok": False, "error": "no spec"}
    add("B2_baseline_resolvable", base_res.get("ok"),
        base_res.get("error") or f"resolved to {base_res.get('qualname')} @ {base_res.get('file')}")

    base_origin = origin_of(base_res.get("file"), task_dir, eval_dir) if base_res.get("ok") else \
        {"kind": "unresolved", "distribution": None, "path": None}
    add("B3_baseline_origin_installed", base_origin["kind"] == "installed",
        f"origin={base_origin['kind']}"
        + (f" dist={base_origin['distribution']}" if base_origin["distribution"] else "")
        + (" -- a baseline living in the task/eval dir is the candidate's own scaffold, not the online kernel"
           if base_origin["kind"] in ("task_local", "eval_local") else "")
        + (" -- source file is not inside any installed distribution" if base_origin["kind"] == "unknown" else ""))

    tgt_res = resolve_spec(tgt_spec) if tgt_spec else {"ok": False, "error": "meta.target_callable missing/empty"}
    add("B4_target_resolvable", tgt_res.get("ok"),
        tgt_res.get("error") or f"resolved to {tgt_res.get('qualname')} @ {tgt_res.get('file')}")

    # B5: the denominator must BE the seam, or be provably reached FROM it. Identity is checked on the
    # resolved code object (so two spellings of the same function pass). Otherwise we require positive
    # evidence recorded at capture time -- an assertion in prose is not evidence.
    same_obj = bool(base_res.get("ok") and tgt_res.get("ok")
                    and (base_res.get("obj") is tgt_res.get("obj")
                         or getattr(base_res.get("obj"), "__code__", None)
                         is getattr(tgt_res.get("obj"), "__code__", object())))
    ev = meta.get("baseline_capture_evidence") or {}
    observed = isinstance(ev, dict) and int(ev.get("observed_calls") or 0) > 0 \
        and (ev.get("from_seam") or "").strip() == tgt_spec
    add("B5_denominator_is_the_seam", same_obj or observed,
        "baseline_callable IS target_callable" if same_obj else
        (f"observed {ev.get('observed_calls')} live call(s) from {ev.get('from_seam')!r}" if observed else
         "baseline_callable is neither the seam nor an OBSERVED callee of it "
         "(set meta.baseline_capture_evidence={from_seam, observed_calls} at capture time)"))

    ok = all(c["ok"] for c in checks)
    return {
        "contract": "baseline_identity", "contract_version": CONTRACT_VERSION, "ok": ok,
        "baseline_callable": base_spec, "target_callable": tgt_spec,
        "baseline_origin": {k: v for k, v in base_origin.items() if k != "obj"},
        "checks": checks,
        "failed": [c["id"] for c in checks if not c["ok"]],
        "verdict": ("denominator is the live online kernel" if ok else
                    "INVALID DENOMINATOR -- any speedup measured against it is not comparable to e2e"),
    }


# -------------------------------------------------------------------------- INV-2 binding contract
def _describe_signature(sig, spec, qualname, resolved_file=None, meta=None):
    """Build a binding descriptor from an already-resolved inspect.Signature."""
    meta = dict(meta or {})
    params, required_positional, out_params = [], [], []
    accepts_varargs = accepts_varkw = False
    for name, p in sig.parameters.items():
        if p.kind is inspect.Parameter.VAR_POSITIONAL:
            accepts_varargs = True
        if p.kind is inspect.Parameter.VAR_KEYWORD:
            accepts_varkw = True
        ann = "" if p.annotation is inspect.Parameter.empty else _ann_str(p.annotation)
        has_def = p.default is not inspect.Parameter.empty
        params.append({"name": name, "kind": p.kind.name, "has_default": has_def, "annotation": ann})
        if (not has_def and p.kind in (inspect.Parameter.POSITIONAL_ONLY,
                                       inspect.Parameter.POSITIONAL_OR_KEYWORD,
                                       inspect.Parameter.KEYWORD_ONLY)):
            required_positional.append(name)
        if name.lower() in _OUT_PARAM_NAMES:
            out_params.append(name)

    ret_ann = "" if sig.return_annotation is inspect.Signature.empty else _ann_str(sig.return_annotation)
    returns_none = ret_ann in ("None", "NoneType")

    # Runtime evidence from the capture step, when the extractor recorded it, beats the name heuristic.
    ev = meta.get("seam_runtime_evidence") or {}
    if isinstance(ev, dict) and ev.get("inplace_params") is not None:
        out_params = list(ev.get("inplace_params") or [])
        evidence = "observed_inplace"
    elif isinstance(ev, dict) and ev.get("returns_none") is not None:
        returns_none = bool(ev.get("returns_none"))
        evidence = "observed_return"
    else:
        evidence = "name_convention" if out_params else "none"

    # Inputs the callable reads that do NOT arrive through its parameters (forward-context, layer
    # registries, module globals). A non-empty list means the seam is NOT a pure function of its
    # arguments, so an out-of-tree rewrite cannot be given the same inputs -- see check_binding.
    hidden_known = isinstance(ev, dict) and isinstance(ev.get("hidden_context"), list)
    hidden_ctx = list(ev["hidden_context"]) if hidden_known else []

    return {
        "contract": "binding", "contract_version": CONTRACT_VERSION, "ok": True,
        "seam": spec, "qualname": qualname, "resolved_file": resolved_file,
        "params": params, "required_positional": required_positional,
        "accepts_varargs": accepts_varargs, "accepts_varkw": accepts_varkw,
        "arity_required": len(required_positional), "arity_total": len(params),
        "returns_annotation": ret_ann, "returns_none": returns_none,
        "out_params": out_params, "out_params_evidence": evidence,
        "hidden_context": hidden_ctx,
        "hidden_context_evidence": "declared" if hidden_known else "unknown",
        "signature": f"{qualname}{sig}",
    }


def describe_binding(spec, meta=None):
    """Capture the LIVE callable's call contract by reflection. This descriptor -- not an agent's
    recollection of it -- is what an authored entry has to satisfy."""
    meta = dict(meta or {})
    res = resolve_spec(spec)
    if not res.get("ok"):
        return {"contract": "binding", "contract_version": CONTRACT_VERSION, "ok": False,
                "seam": spec, "error": res.get("error")}
    obj = res["obj"]
    if not callable(obj):
        return {"contract": "binding", "contract_version": CONTRACT_VERSION, "ok": False,
                "seam": spec, "error": "seam resolves to a non-callable"}
    try:
        sig = inspect.signature(obj)
    except Exception as e:  # noqa: BLE001
        return {"contract": "binding", "contract_version": CONTRACT_VERSION, "ok": False,
                "seam": spec, "error": f"signature unavailable: {e!r}"}
    return _describe_signature(sig, spec, res.get("qualname"), res.get("file"), meta)


def _ann_str(ann):
    if ann is None:
        return "None"
    if isinstance(ann, str):
        return ann
    return getattr(ann, "__name__", None) or str(ann)


def _signature_from_descriptor(desc):
    """Reconstruct an inspect.Signature for call-compatibility checks."""
    params = []
    for p in desc.get("params") or []:
        kind = getattr(inspect.Parameter, p["kind"])
        default = None if p.get("has_default") else inspect.Parameter.empty
        params.append(inspect.Parameter(p["name"], kind, default=default))
    return inspect.Signature(params)


def _representative_calls(desc):
    """Calls spanning the live signature's positional/keyword and optional surfaces."""
    calls = []
    for include_optional in (False, True):
        for keyword_pok in (False, True):
            args, kwargs = [], {}
            for p in desc.get("params") or []:
                kind, name = p["kind"], p["name"]
                required = not p.get("has_default") and kind not in ("VAR_POSITIONAL", "VAR_KEYWORD")
                if kind == "VAR_POSITIONAL":
                    if include_optional:
                        args.append(object())
                    continue
                if kind == "VAR_KEYWORD":
                    if include_optional:
                        kwargs["__geak_extra_kwarg__"] = object()
                    continue
                if not required and not include_optional:
                    continue
                value = object()
                if kind == "POSITIONAL_ONLY":
                    args.append(value)
                elif kind == "POSITIONAL_OR_KEYWORD":
                    if keyword_pok:
                        kwargs[name] = value
                    else:
                        args.append(value)
                elif kind == "KEYWORD_ONLY":
                    kwargs[name] = value
            calls.append((args, kwargs))
    return calls


def check_binding(descriptor, candidate):
    """Can `candidate` be bound AT the seam described by `descriptor`?

    `candidate` may be a module:attr spec, an already-built descriptor, or a dict
    {params, returns_none, ...}. Mismatch codes are a closed set so the orchestrator can route on them
    instead of pattern-matching prose.
    """
    if not descriptor or not descriptor.get("ok"):
        return {"contract": "binding_check", "contract_version": CONTRACT_VERSION, "bindable": False,
                "seam": (descriptor or {}).get("seam", ""), "candidate": str(candidate),
                "mismatches": [{"code": "no_seam_descriptor",
                                "detail": (descriptor or {}).get("error", "seam was never described")}],
                "codes": ["no_seam_descriptor"]}

    if isinstance(candidate, str):
        cand = describe_binding(candidate)
        if not cand.get("ok"):
            return {"contract": "binding_check", "contract_version": CONTRACT_VERSION, "bindable": False,
                    "seam": descriptor.get("seam"), "candidate": candidate,
                    "mismatches": [{"code": "candidate_unresolvable", "detail": cand.get("error")}],
                    "codes": ["candidate_unresolvable"]}
    else:
        cand = dict(candidate or {})

    m = []
    live_req = list(descriptor.get("required_positional") or [])
    cand_req = list(cand.get("required_positional") or [])
    cand_varargs = bool(cand.get("accepts_varargs"))
    cand_varkw = bool(cand.get("accepts_varkw"))

    # Arity. A single opaque `args`/`kwargs`-style parameter standing in for N live tensors is the
    # classic out-of-tree rewrite that can never be rebound; it shows up here as an arity mismatch.
    if not cand_varargs and len(cand_req) != len(live_req):
        m.append({"code": "arity_mismatch",
                  "detail": f"seam requires {len(live_req)} positional arg(s) {live_req}; "
                            f"candidate entry requires {len(cand_req)} {cand_req}"})

    # Names. The overlay rebinds by NAME at keyword call sites, so a renamed required parameter is a
    # hard break even when the arity lines up.
    if not (cand_varargs and cand_varkw):
        missing = [n for n in live_req if n not in [p["name"] for p in (cand.get("params") or [])]]
        if missing and not cand_varkw:
            m.append({"code": "param_name_mismatch",
                      "detail": f"seam parameter(s) {missing} absent from the candidate entry"})

    # Return contract. A seam that writes into a caller-owned buffer and returns None cannot be
    # replaced by something that allocates and returns a fresh tensor -- the caller never reads it.
    live_inplace = bool(descriptor.get("out_params")) or bool(descriptor.get("returns_none"))
    cand_inplace = bool(cand.get("out_params")) or bool(cand.get("returns_none"))
    if live_inplace and not cand_inplace:
        m.append({"code": "return_contract_mismatch",
                  "detail": f"seam writes in place (out_params={descriptor.get('out_params')}, "
                            f"returns_none={descriptor.get('returns_none')}) but the candidate returns a "
                            f"fresh value; the live caller would discard the result"})
    if cand_inplace and not live_inplace:
        m.append({"code": "return_contract_mismatch",
                  "detail": "candidate writes in place but the seam's callers consume a returned value"})

    # C6: an OPTIONAL live parameter that the candidate DROPS is still passed by existing callers, so the
    # bound call raises TypeError even though the REQUIRED names+arity line up. (Comparing only the
    # required-name sets missed this.) A candidate that swallows extras via **kwargs is exempt.
    if not cand_varkw:
        cand_names = {p["name"] for p in (cand.get("params") or [])}
        live_optional = [p["name"] for p in (descriptor.get("params") or [])
                         if p.get("has_default") and p["kind"] in ("POSITIONAL_OR_KEYWORD", "KEYWORD_ONLY")]
        dropped = [n for n in live_optional if n not in cand_names]
        if dropped:
            m.append({"code": "optional_param_dropped",
                      "detail": f"seam accepts optional param(s) {dropped} that live callers may pass; the "
                                f"candidate entry omits them and would raise TypeError when they are"})

    # Check actual Python binding behavior over the live signature's minimal/maximal and
    # positional/keyword call surfaces. This catches positional->keyword-only changes, reordered
    # positional parameters when keyword calls are also legal, and variadic incompatibilities without
    # incorrectly rejecting a live POSITIONAL_ONLY parameter against an identical candidate.
    try:
        cand_sig = _signature_from_descriptor(cand)
        for call_args, call_kwargs in _representative_calls(descriptor):
            try:
                cand_sig.bind(*call_args, **call_kwargs)
            except TypeError as e:
                if not any(x["code"] in ("arity_mismatch", "param_name_mismatch",
                                         "optional_param_dropped", "param_kind_mismatch") for x in m):
                    m.append({"code": "param_kind_mismatch",
                              "detail": f"candidate rejects a call accepted by the live seam: {e}"})
                break
    except (TypeError, ValueError, KeyError) as e:
        m.append({"code": "param_kind_mismatch",
                  "detail": f"candidate signature descriptor is invalid: {e}"})

    # Hidden context. If the seam reads state that never crosses the parameter boundary, an authored
    # replacement cannot be handed the same inputs -- authoring it is wasted budget regardless of how
    # fast it is. This is the check that costs seconds and saves a kernel budget.
    if descriptor.get("hidden_context_evidence") != "declared":
        m.append({"code": "hidden_context_inputs",
                  "detail": "seam_runtime_evidence.hidden_context is missing; purity is unknown, so the "
                            "seam cannot be admitted under the fail-closed binding contract"})
    elif descriptor.get("hidden_context"):
        m.append({"code": "hidden_context_inputs",
                  "detail": f"seam reads non-parameter inputs {descriptor['hidden_context']}; it is not a "
                            f"pure function of its arguments -- rebind at an inner seam that is"})

    return {"contract": "binding_check", "contract_version": CONTRACT_VERSION,
            "bindable": not m, "seam": descriptor.get("seam"),
            "candidate": cand.get("seam") or cand.get("qualname") or "<entry>",
            "mismatches": m, "codes": [x["code"] for x in m]}


def render_entry(descriptor, entry_name="entry"):
    """Emit the unittest entry contract FROM the live signature, so an authored kernel is written
    against the real call shape instead of one the extractor invented. Generated, never hand-written:
    that is what makes 'signature mismatch' structurally impossible rather than merely detected."""
    if not descriptor or not descriptor.get("ok"):
        raise ValueError("cannot render an entry from a failed binding descriptor")
    parts = []
    for p in descriptor.get("params") or []:
        k, n = p["kind"], p["name"]
        if k == "VAR_POSITIONAL":
            parts.append(f"*{n}")
        elif k == "VAR_KEYWORD":
            parts.append(f"**{n}")
        elif p["has_default"]:
            parts.append(f"{n}=None")
        else:
            parts.append(n)
    posonly = [p["name"] for p in (descriptor.get("params") or []) if p["kind"] == "POSITIONAL_ONLY"]
    if posonly:
        last = posonly[-1]
        i = parts.index(last) if last in parts else parts.index(f"{last}=None")
        parts.insert(i + 1, "/")
    has_star = any(p["kind"] == "VAR_POSITIONAL" for p in descriptor.get("params") or [])
    kwonly = [p["name"] for p in (descriptor.get("params") or []) if p["kind"] == "KEYWORD_ONLY"]
    if kwonly and not has_star:
        i = min(parts.index(n) if n in parts else parts.index(f"{n}=None") for n in kwonly)
        parts.insert(i, "*")
    sig = ", ".join(parts)
    out = descriptor.get("out_params") or []
    ret = ("    # The live seam writes into %s and returns None -- write there, return None.\n"
           "    raise NotImplementedError\n" % (out,)) if (out or descriptor.get("returns_none")) else \
          "    # The live seam returns its result -- return it.\n    raise NotImplementedError\n"
    return (
        '"""AUTO-GENERATED from the live seam by seam_contract.render_entry -- DO NOT EDIT BY HAND.\n'
        f'Seam: {descriptor.get("seam")}\n'
        f'Live signature: {descriptor.get("signature")}\n'
        'Any authored kernel MUST implement exactly this contract; the overlay rebinds this name.\n'
        '"""\n\n'
        f"CONTRACT_SEAM = {descriptor.get('seam')!r}\n"
        f"CONTRACT_VERSION = {CONTRACT_VERSION}\n\n\n"
        f"def {entry_name}({sig}):\n{ret}")


# ------------------------------------------------------------------------------------------- CLI
def _load_meta(task_dir):
    p = os.path.join(task_dir, "meta.json")
    if not os.path.exists(p):
        return {}, f"no meta.json in {task_dir}"
    try:
        with open(p) as fh:
            return json.load(fh), None
    except Exception as e:  # noqa: BLE001
        return {}, f"meta.json unreadable: {e!r}"


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--task-dir", default="", help="op task dir containing meta.json")
    ap.add_argument("--eval-dir", default="", help="run EVAL_DIR (also disqualified as a baseline origin)")
    # C10: the binding DESCRIPTOR must be built from the deployment TARGET, never the baseline. Two
    # explicit flags make the intent unambiguous; --spec stays as a DEPRECATED alias for --target-spec so
    # existing callers keep working (it used to feed the descriptor and, when a role piped the
    # baseline_callable through it, silently described the denominator instead of the deployment seam).
    ap.add_argument("--target-spec", default="", help="module:attr of the DEPLOYMENT seam to describe (the binding target)")
    ap.add_argument("--baseline-spec", default="", help="module:attr of the baseline/denominator (overrides meta.baseline_callable for validation)")
    ap.add_argument("--spec", default="", help="DEPRECATED alias for --target-spec")
    ap.add_argument("--mode", default="both", choices=["baseline", "binding", "both", "entry"])
    ap.add_argument("--candidate", default="", help="module:attr of the authored entry, for --mode binding")
    ap.add_argument("--entry-name", default="entry")
    ap.add_argument("--out", default="", help="write the rendered entry contract here (--mode entry)")
    ap.add_argument("--json", action="store_true", help="print the verdict as JSON (default)")
    ap.add_argument("--site-root", action="append", default=[],
                    help="extra install root to count as 'installed' (overlay/editable/vendored); repeatable")
    args = ap.parse_args(argv)

    if args.site_root:
        prev = os.environ.get("GEAK_SEAM_SITE_DIRS", "")
        os.environ["GEAK_SEAM_SITE_DIRS"] = os.pathsep.join([p for p in ([prev] + args.site_root) if p])

    meta, meta_err = ({}, None)
    if args.task_dir:
        meta, meta_err = _load_meta(args.task_dir)

    # Explicit CLI specs override the corresponding meta fields for this validation run.
    if args.baseline_spec:
        meta = dict(meta)
        meta["baseline_callable"] = args.baseline_spec
    if args.target_spec or args.spec:
        meta = dict(meta)
        meta["target_callable"] = args.target_spec or args.spec

    result = {"contract_version": CONTRACT_VERSION, "task_dir": args.task_dir or None}
    if meta_err:
        result["meta_error"] = meta_err
    if args.spec and not args.target_spec:
        sys.stderr.write("seam_contract: --spec is deprecated; use --target-spec for the deployment seam "
                         "(--spec builds the binding DESCRIPTOR, never the baseline)\n")

    if args.mode in ("baseline", "both"):
        result["baseline_validation"] = validate_baseline(args.task_dir or None, meta, args.eval_dir or None)

    if args.mode in ("binding", "both", "entry"):
        # C10: the descriptor is the DEPLOYMENT target's contract. Precedence: --target-spec, then the
        # deprecated --spec alias, then meta.target_callable. The baseline is NEVER what gets described.
        spec = args.target_spec or args.spec or (meta.get("target_callable") or "")
        desc = describe_binding(spec, meta) if spec else {
            "contract": "binding", "contract_version": CONTRACT_VERSION, "ok": False,
            "seam": "", "error": "no target_callable / --target-spec given"}
        result["binding_descriptor"] = desc
        if args.candidate:
            result["binding_check"] = check_binding(desc, args.candidate)
        elif desc.get("ok") and args.mode in ("binding", "both"):
            # Validate the ACTUAL signature emitted by render_entry(), not the descriptor against itself.
            # This proves the generated immutable entry contract accepts every live call shape.
            try:
                ns = {}
                exec(render_entry(desc, args.entry_name), ns)  # generated source; no untrusted input runs
                generated = ns[args.entry_name]
                generated_meta = {"seam_runtime_evidence": {
                    "inplace_params": list(desc.get("out_params") or []),
                    "returns_none": bool(desc.get("returns_none")),
                    "hidden_context": [],
                }}
                cand_desc = _describe_signature(inspect.signature(generated),
                                                "<rendered_entry>",
                                                args.entry_name, None, generated_meta)
                result["binding_check"] = check_binding(desc, cand_desc)
                result["binding_check"]["candidate"] = "<rendered_entry>"
            except Exception as e:  # noqa: BLE001
                result["binding_check"] = {
                    "contract": "binding_check", "contract_version": CONTRACT_VERSION,
                    "bindable": False, "seam": desc.get("seam"), "candidate": "<rendered_entry>",
                    "mismatches": [{"code": "candidate_unresolvable",
                                    "detail": f"generated entry could not be inspected: {e!r}"}],
                    "codes": ["candidate_unresolvable"],
                }
        if args.mode == "entry":
            if not desc.get("ok"):
                result["entry_error"] = desc.get("error")
            else:
                src = render_entry(desc, args.entry_name)
                result["entry_contract"] = src
                if args.out:
                    with open(args.out, "w") as fh:
                        fh.write(src)
                    result["entry_path"] = os.path.realpath(args.out)

    ok = True
    if "baseline_validation" in result:
        ok = ok and bool(result["baseline_validation"].get("ok"))
    if args.mode in ("binding", "entry") or ("binding_check" in result):
        ok = ok and bool(result.get("binding_descriptor", {}).get("ok"))
        if "binding_check" in result:
            ok = ok and bool(result["binding_check"].get("bindable"))
    result["ok"] = ok

    print(json.dumps(result, indent=2, default=str))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
