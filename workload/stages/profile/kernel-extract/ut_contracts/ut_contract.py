#!/usr/bin/env python3
"""Per-operator UT contracts: declare once what a kernel computes, check every task against it.

A family contract (`families/<family>.json`) states, for one operator family, what a unit test
must hold fixed and what it may vary: the axes and their legal ranges, the attributes that select
a variant, the constraints between them, every input's shape, dtype, value distribution, lifetime
and stored layout, and every output's error bound. `scaffold_ut.py` freezes the family contract
plus the task's own choices into `<task>/run/ut_contract.json`; this module is copied into the
task beside `harness_lib.py` and reads only that frozen copy.

What it gives a task, in the order a UT needs it:

  lint_task(task)             CPU-only: contract, case manifest and meta agree (no GPU, no torch)
  build_inputs(...)           seeded operands drawn from the declared distributions
  check_inputs(...)           construction is reproducible, layout facts hold, values are legal
  compare_outputs(...)        candidate vs baseline (and vs a float64 reference) under the
                              output's declared bound, not one task-wide `tol`

The design follows SIKL's OpContract (sikl/core/contract): a bound names the arithmetic it
tolerates (a dot product of length K, a rounding to the output dtype, a baseline's own error), an
integer output is exact, a top-k output may order ties freely, and an input's distribution is part
of the contract rather than whatever the author drew. torch is imported lazily so the lint runs on
a login host.
"""

import ast
from collections import OrderedDict
import hashlib
import json
import math
import os
import re
import sys

CONTRACT_VERSION = 1
HERE = os.path.dirname(os.path.abspath(__file__))
FAMILIES_DIR = os.path.join(HERE, "families")

BOUND_KINDS = ("exact", "multiset_rows", "topk_equivalent", "tolerance", "dot_product",
               "output_rounding", "baseline_relative", "noise_relative", "sqnr", "bytes_exact",
               "harness")
BOUNDS_NEEDING_REFERENCE = ("dot_product", "baseline_relative", "output_rounding")
# `reference: "oracle:<name>"` names a leg of `oracles` (another launch path of the same step,
# e.g. a monokernel's per-stage launches) instead of a float64 function in reference_f64.py.
ORACLE_PREFIX = "oracle:"
ORACLE_CASES = ("all", "per_width_smallest")
DIST_KINDS = ("normal", "lecun_normal", "uniform", "log_uniform", "int_range", "permutation",
              "routing", "simplex", "cu_seqlens", "seq_lens", "page_table", "sparse_indices",
              "constant", "fixture", "hook", "produced")
LIFETIMES = ("weight", "per_call", "state")
# Weight-model sources the timing gate accepts for an AgentX task; keep in step with
# harness_lib._observed_workload_cases.
OBSERVED_WEIGHT_SOURCES = ("observed_workload", "agentx_log_fold")
DETERMINISM = ("bitwise", "bounded")

# float64 unit roundoff (half an ulp at 1.0) of each output dtype a bound may round to.
UNIT_ROUNDOFF = {"float64": 2.0 ** -53, "float32": 2.0 ** -24, "bfloat16": 2.0 ** -8,
                 "float16": 2.0 ** -11, "float8_e4m3fn": 2.0 ** -4, "float8_e4m3fnuz": 2.0 ** -4,
                 "float8_e5m2": 2.0 ** -3, "float8_e5m2fnuz": 2.0 ** -3}
INTEGER_DTYPES = ("int8", "int16", "int32", "int64", "uint8", "bool")
# Packed or exponent-only dtypes: a tensor of these is compared byte for byte or decoded first.
PACKED_DTYPES = ("float4_e2m1fn_x2", "float8_e8m0fnu", "uint8")

# Elementwise bound multiples, as in SIKL semantics/_compare.py.
_SQRT_LENGTH_MULTIPLE = 6.0
_CONSTANT_MULTIPLE = 2.0
_SLACK = 2.0
_OUTPUT_ROUNDING_MULTIPLE = 2.0
_MAX_ERROR_MULTIPLE = 2.0


class ContractError(ValueError):
    """The contract, the task's choices or a case contradict each other."""


# ----------------------------------------------------------------------------- expressions
_ALLOWED_FUNCS = {
    "ceil_div": lambda a, b: -(-a // b),
    "min": min, "max": max, "abs": abs,
    "bit_length": lambda a: int(a).bit_length(),
    "round_up": lambda a, b: -(-a // b) * b,
}
_ALLOWED_NODES = (ast.Expression, ast.BinOp, ast.UnaryOp, ast.BoolOp, ast.Compare, ast.Name,
                  ast.Load, ast.Constant, ast.Call, ast.IfExp, ast.Add, ast.Sub, ast.Mult,
                  ast.FloorDiv, ast.Div, ast.Mod, ast.Pow, ast.USub, ast.UAdd, ast.Not, ast.And,
                  ast.Or, ast.Eq, ast.NotEq, ast.Lt, ast.LtE, ast.Gt, ast.GtE, ast.In, ast.NotIn,
                  ast.Tuple, ast.List)


def evaluate(expr, symbols):
    """Evaluate a contract expression (`"ceil_div(k, 128)"`, `"topk <= n_experts"`) over
    `symbols`. Only arithmetic, comparisons, boolean operators and `_ALLOWED_FUNCS` are legal;
    an int or bool passes through."""
    if isinstance(expr, (bool, int, float)):
        return expr
    tree = ast.parse(str(expr), mode="eval")
    for node in ast.walk(tree):
        if not isinstance(node, _ALLOWED_NODES):
            raise ContractError("illegal construct %s in %r" % (type(node).__name__, expr))
        if isinstance(node, ast.Call) and not (isinstance(node.func, ast.Name)
                                               and node.func.id in _ALLOWED_FUNCS):
            raise ContractError("illegal call in %r" % expr)
        if isinstance(node, ast.Name) and node.id not in symbols and node.id not in _ALLOWED_FUNCS \
                and node.id not in ("True", "False", "None"):
            raise ContractError("unknown symbol %r in %r" % (node.id, expr))
    return eval(compile(tree, "<contract>", "eval"), {"__builtins__": {}},
                {**_ALLOWED_FUNCS, **symbols})


# ----------------------------------------------------------------------------- loading
def load_family(family, families_dir=FAMILIES_DIR):
    path = os.path.join(families_dir, family + ".json")
    if not os.path.exists(path):
        known = sorted(f[:-5] for f in os.listdir(families_dir) if f.endswith(".json"))
        raise ContractError("no family %r; known: %s" % (family, ", ".join(known)))
    with open(path) as f:
        doc = json.load(f)
    validate_family(doc)
    return doc


def load_task_contract(task_dir):
    """The frozen contract a task was scaffolded with: `run/ut_contract.json`."""
    path = os.path.join(task_dir, "run", "ut_contract.json")
    with open(path) as f:
        doc = json.load(f)
    validate_family(doc)
    return doc


def contract_sha256(doc):
    return hashlib.sha256(json.dumps(doc, sort_keys=True).encode()).hexdigest()


def validate_family(doc):
    """Structural check of a contract document; raises ContractError on the first defect."""
    need = ("ut_contract_version", "family", "op_kind", "description", "axes", "inputs", "outputs",
            "compare")
    missing = [k for k in need if k not in doc]
    if missing:
        raise ContractError("contract lacks %s" % missing)
    if doc["ut_contract_version"] != CONTRACT_VERSION:
        raise ContractError("contract version %r, this library reads %d"
                            % (doc["ut_contract_version"], CONTRACT_VERSION))
    generation = doc.get("input_generation")
    if generation is not None:
        budget = generation.get("weight_cache_bytes", 2 << 30)
        if generation.get("protocol") != "weight_substreams/v1" or type(budget) is not int or budget < 0:
            raise ContractError("invalid input_generation protocol or weight cache budget")
    names = list(doc["axes"]) + list(doc.get("attrs", {})) + list(doc["inputs"]) + list(doc["outputs"])
    dup = sorted({n for n in names if names.count(n) > 1})
    if dup:
        raise ContractError("names used twice: %s" % dup)
    for name, a in doc["axes"].items():
        if "expr" in a:
            continue
        if not ("lo" in a and "hi" in a) or a["lo"] > a["hi"]:
            raise ContractError("axis %s needs lo <= hi (or an expr)" % name)
    for name, a in doc.get("attrs", {}).items():
        if not a.get("values"):
            raise ContractError("attr %s needs a non-empty values list" % name)
    for name, s in doc["inputs"].items():
        if s.get("kind", "tensor") == "scalar":
            continue
        if "shape" not in s or "dtype" not in s:
            raise ContractError("input %s needs shape and dtype" % name)
        if (s.get("dist") or {}).get("kind") == "produced" and s["dist"].get("by") not in doc["inputs"]:
            raise ContractError("input %s: produced by unknown input %r" % (name, s["dist"].get("by")))
        dist = s.get("dist") or {}
        if dist.get("kind") not in DIST_KINDS:
            raise ContractError("input %s: distribution kind %r not in %s"
                                % (name, dist.get("kind"), DIST_KINDS))
        if s.get("lifetime", "per_call") not in LIFETIMES:
            raise ContractError("input %s: lifetime %r" % (name, s.get("lifetime")))
    for name, s in doc["outputs"].items():
        bound = s.get("bound") or {}
        if bound.get("kind") not in BOUND_KINDS:
            raise ContractError("output %s: bound kind %r not in %s"
                                % (name, bound.get("kind"), BOUND_KINDS))
        if bound["kind"] == "topk_equivalent":
            src = bound.get("scores") or ""
            if not (src in doc["inputs"] or (src.startswith("ref:") and f64_reference(doc))):
                raise ContractError("output %s: topk_equivalent scores must name an input or "
                                    "ref:<output> of a contract with a float64 reference" % name)
        if bound["kind"] in BOUNDS_NEEDING_REFERENCE and not f64_reference(doc):
            # an oracle leg yields outputs, not the float64 values / magnitudes these bounds read
            raise ContractError("output %s: bound %s needs a float64 reference (an oracle:<name> "
                                "reference does not provide one) and the contract names none"
                                % (name, bound["kind"]))
        _validate_delta(doc, name, bound)
        if not isinstance(s.get("require_finite", False), bool):
            raise ContractError("output %s: require_finite must be true or false" % name)
        opt = s.get("optional_if_not_produced")
        if opt is not None:
            if not isinstance(opt, dict) or isinstance(opt.get("poison"), bool) \
                    or not isinstance(opt.get("poison"), (int, float)):
                raise ContractError("output %s: optional_if_not_produced needs {\"poison\": <number>}, "
                                    "the value the harness fills the buffer with before the call" % name)
            if bound["kind"] not in ("exact", "bytes_exact"):
                raise ContractError("output %s: optional_if_not_produced is for exact outputs; a "
                                    "poison value inside a tolerance cannot be told from a result" % name)
    _validate_oracles(doc)
    _validate_control_categories(doc)
    if doc["compare"].get("determinism", "bounded") not in DETERMINISM:
        raise ContractError("compare.determinism %r" % doc["compare"].get("determinism"))
    return doc


def _validate_delta(doc, name, bound, where="output"):
    """`bound.delta: {"minus": <output>, "along"?: <axis>}` judges the step's change, not the
    state: `out - minus`, or with `along` the consecutive differences of `out` over that axis
    (the first against `minus`). For residual-stream outputs, whose pass-through residual
    otherwise dominates any relative bound."""
    d = bound.get("delta")
    if d is None:
        return
    if not isinstance(d, dict) or d.get("minus") not in doc["outputs"] or d.get("minus") == name:
        raise ContractError("%s %s: delta.minus must name another output of the contract" % (where, name))
    if bound["kind"] not in ("tolerance", "sqnr"):
        raise ContractError("%s %s: delta applies to tolerance or sqnr bounds" % (where, name))
    if "along" in d:
        shape = doc["outputs"][name].get("shape")
        if not isinstance(shape, list) or d["along"] not in shape:
            raise ContractError("%s %s: delta.along %r is not an axis of the output's shape"
                                % (where, name, d["along"]))
    if f64_reference(doc):
        raise ContractError("%s %s: delta is for oracle-referenced contracts (no float64 "
                            "reference of the differenced output)" % (where, name))


def _validate_oracles(doc):
    """`oracles` are named legs that recompute the step another way; `reference: "oracle:<name>"`
    makes one of them what every output is compared against. An oracle may narrow the outputs
    it produces and carry its own bounds (a looser one for an independent reimplementation)."""
    oracles = doc.get("oracles")
    if oracles is not None and not isinstance(oracles, dict):
        raise ContractError("oracles must map a name to its leg")
    oracles = oracles or {}
    ref = doc.get("reference")
    if isinstance(ref, str) and ref.startswith(ORACLE_PREFIX):
        name = ref[len(ORACLE_PREFIX):]
        if name not in oracles:
            raise ContractError("reference %r names no oracle; declared: %s" % (ref, sorted(oracles)))
    for name, o in oracles.items():
        if not isinstance(o, dict) or not isinstance(o.get("path"), str) or not o["path"]:
            raise ContractError("oracle %s needs a path (the cases.ORACLES entry it runs)" % name)
        outs = o.get("outputs")
        if outs is not None:
            if not isinstance(outs, list) or not outs:
                raise ContractError("oracle %s: outputs must be a non-empty list" % name)
            extra = sorted(set(outs) - set(doc["outputs"]))
            if extra:
                raise ContractError("oracle %s: outputs %s are not outputs of the contract" % (name, extra))
        allowed = set(outs if outs is not None else doc["outputs"])
        for out, bound in (o.get("bounds") or {}).items():
            if out not in allowed:
                raise ContractError("oracle %s: bound for %s, which the oracle does not produce" % (name, out))
            if not isinstance(bound, dict) or bound.get("kind") not in BOUND_KINDS:
                raise ContractError("oracle %s: output %s: bound kind %r not in %s"
                                    % (name, out, (bound or {}).get("kind"), BOUND_KINDS))
            if bound["kind"] == "topk_equivalent":
                raise ContractError("oracle %s: topk_equivalent ranks scores, not an oracle leg" % name)
            if bound["kind"] in BOUNDS_NEEDING_REFERENCE:
                raise ContractError("oracle %s: output %s: bound %s needs float64 reference values "
                                    "an oracle leg does not produce" % (name, out, bound["kind"]))
            _validate_delta(doc, out, bound, where="oracle %s: output" % name)
            minus = (bound.get("delta") or {}).get("minus")
            if minus is not None and minus not in allowed:
                raise ContractError("oracle %s: output %s: delta.minus %s is not an output of the oracle"
                                    % (name, out, minus))
        if not isinstance(o.get("controls", True), bool):
            raise ContractError("oracle %s: controls must be true or false" % name)
        if o.get("cases", "all") not in ORACLE_CASES:
            raise ContractError("oracle %s: cases %r not in %s" % (name, o.get("cases"), ORACLE_CASES))


def _validate_control_categories(doc):
    """`required_control_categories`: the kinds of injected bug a task's negative controls must
    cover, `{"by": <attr>, "cases": {<attr value>: [category, ...]}}` or a plain list."""
    req = doc.get("required_control_categories")
    if req is None or isinstance(req, list) and all(isinstance(c, str) for c in req):
        return
    if not isinstance(req, dict) or not isinstance(req.get("cases"), dict):
        raise ContractError("required_control_categories must be a list or {by, cases}")
    attr = (doc.get("attrs") or {}).get(req.get("by"))
    if attr is None:
        raise ContractError("required_control_categories.by %r is not an attr" % req.get("by"))
    if not isinstance(attr, dict) or not isinstance(attr.get("values"), list):
        raise ContractError("required_control_categories.by %r: the attr declares no values list"
                            % req.get("by"))
    for value, cats in req["cases"].items():
        if value not in [str(v) for v in attr["values"]]:
            raise ContractError("required_control_categories: %s=%r not in %s"
                                % (req["by"], value, attr["values"]))
        if not isinstance(cats, list) or not cats or not all(isinstance(c, str) for c in cats):
            raise ContractError("required_control_categories[%s] must be a non-empty list" % value)


def f64_reference(doc):
    """The reference_f64.py function a contract names, or None (no reference, or an oracle leg)."""
    ref = doc.get("reference")
    if not ref or (isinstance(ref, str) and ref.startswith(ORACLE_PREFIX)):
        return None
    return ref


def reference_oracle(doc):
    """The oracle `reference: "oracle:<name>"` names, or None."""
    ref = doc.get("reference")
    if isinstance(ref, str) and ref.startswith(ORACLE_PREFIX):
        return ref[len(ORACLE_PREFIX):]
    return None


def oracle_doc(doc, name):
    """The contract one oracle leg is judged under: outputs narrowed to what the oracle produces,
    each output's bound replaced by the oracle's own where it declares one."""
    o = doc["oracles"][name]
    out = json.loads(json.dumps(doc))
    keep = o.get("outputs") or list(doc["outputs"])
    out["outputs"] = {n: s for n, s in out["outputs"].items() if n in keep}
    for n, bound in (o.get("bounds") or {}).items():
        out["outputs"][n]["bound"] = dict(bound)
    return out


def control_categories(doc, sym):
    """The negative-control categories a task with these attrs must cover. A task whose attrs do
    not select one of the declared cases is an error, never "no categories required"."""
    req = doc.get("required_control_categories")
    if not req:
        return []
    if isinstance(req, list):
        return list(req)
    if req["by"] not in (sym or {}):
        raise ContractError("required_control_categories: the task pins no %s" % req["by"])
    key = sym[req["by"]]
    key = str(key).lower() if isinstance(key, bool) else str(key)
    if key not in req["cases"]:
        raise ContractError("required_control_categories: %s=%r has no case; declared: %s"
                            % (req["by"], key, sorted(req["cases"])))
    return list(req["cases"][key])


def freeze_task_contract(family_doc, *, attrs, const_axes, overrides=None):
    """Narrow a family to one task: pin attrs and const axes, apply justified overrides.

    An override is `{"path": "outputs.out.bound", "value": {...}, "why": "..."}`; one without a
    `why` is refused. This is the only way a task departs from its family, and it stays visible."""
    doc = json.loads(json.dumps(family_doc))
    for name, value in attrs.items():
        spec = doc.get("attrs", {}).get(name)
        if spec is None:
            raise ContractError("unknown attr %s" % name)
        if value not in spec["values"]:
            raise ContractError("attr %s=%r not in %s" % (name, value, spec["values"]))
    for name in doc.get("attrs", {}):
        if name not in attrs:
            default = doc["attrs"][name].get("default", doc["attrs"][name]["values"][0])
            attrs = {**attrs, name: default}
    for name, value in const_axes.items():
        a = doc["axes"].get(name)
        if a is None or a.get("var") or "expr" in a:
            raise ContractError("%s is not a const axis of %s" % (name, doc["family"]))
        _check_axis(name, a, value)
    applied = []
    for o in overrides or []:
        if not (o.get("why") or "").strip():
            raise ContractError("override of %s carries no why" % o.get("path"))
        node, keys = doc, o["path"].split(".")
        for k in keys[:-1]:
            node = node[k]
        node[keys[-1]] = o["value"]
        applied.append(o)
    doc["task"] = {"attrs": attrs, "const_axes": const_axes, "overrides": applied,
                   "family_sha256": contract_sha256(family_doc)}
    validate_family(doc)
    return doc


def _check_axis(name, a, value):
    if not isinstance(value, int) or isinstance(value, bool):
        raise ContractError("axis %s=%r is not an int" % (name, value))
    if not a["lo"] <= value <= a["hi"]:
        raise ContractError("axis %s=%d outside [%d, %d]" % (name, value, a["lo"], a["hi"]))
    if a.get("multiple_of") and value % a["multiple_of"]:
        raise ContractError("axis %s=%d not a multiple of %d" % (name, value, a["multiple_of"]))


def symbols_for(doc, case_axes):
    """Every axis (const, var from the case, derived) and attr of one case, checked."""
    task = doc.get("task") or {}
    sym = dict(task.get("attrs", {}))
    sym.update(task.get("const_axes", {}))
    for name, a in doc["axes"].items():
        if "expr" in a:
            continue
        if a.get("var"):
            if name not in case_axes:
                if a.get("optional"):
                    continue
                raise ContractError("case lacks var axis %s" % name)
            _check_axis(name, a, case_axes[name])
            sym[name] = case_axes[name]
        elif name not in sym:
            if name in case_axes:          # a const axis may also be stated per case; must agree
                _check_axis(name, a, case_axes[name])
                sym[name] = case_axes[name]
            elif not a.get("optional"):
                raise ContractError("const axis %s is pinned neither by the task nor the case" % name)
        elif name in case_axes and case_axes[name] != sym[name]:
            raise ContractError("case says %s=%r, task pins %r" % (name, case_axes[name], sym[name]))
    for _ in range(len(doc["axes"])):        # derived axes may build on each other
        for name, a in doc["axes"].items():
            if "expr" in a and name not in sym:
                try:
                    sym[name] = int(evaluate(a["expr"], sym))
                except ContractError:
                    pass
    for c in doc.get("constraints", []):
        if not evaluate(c, sym):
            raise ContractError("constraint violated: %s with %s" % (c, _brief(sym)))
    return sym


def _brief(sym):
    return ", ".join("%s=%s" % (k, v) for k, v in sorted(sym.items()) if not isinstance(v, dict))


def present(spec, sym):
    when = spec.get("present_when")
    return True if when is None else bool(evaluate(when, sym))


def shape_of(spec, sym):
    """`shape` is a list of axis expressions, or `{"by": <attr>, "cases": {<value>: [...]}}`
    when an attribute changes the layout (`trans_b`: (n, k) or (k, n))."""
    shape = spec["shape"]
    if isinstance(shape, dict):
        key = sym[shape["by"]]
        key = str(key).lower() if isinstance(key, bool) else str(key)
        if key not in shape["cases"]:
            raise ContractError("shape has no case %s=%s" % (shape["by"], key))
        shape = shape["cases"][key]
    return tuple(int(evaluate(d, sym)) for d in shape)


def dtype_of(spec, sym):
    """An input's dtype: a single string, a list (first one wins unless the task pins
    `dtype:<name>` as an attr), or `attr:<name>`."""
    d = spec["dtype"]
    if isinstance(d, str) and d.startswith("attr:"):
        return sym[d[5:]]
    if isinstance(d, list):
        return d[0]
    return d


# ----------------------------------------------------------------------------- lint (CPU only)
# Frameworks whose entries need meta.server_preimports, with the module known to rebind
# library functions at import (None: not yet established for that framework).
SERVING_FRAMEWORKS = {"sglang": "sglang.srt.utils.common", "vllm": None}


def _lint_rank_chain(task_dir, doc, meta):
    """A family timed by shared/rank_timing.py (timing.kind rank_chain): the task's meta must
    route timing there for the deployment's rank count, and cases.py must define a filled-in
    rank_chain(). Text checks only: the lint never imports cases.py."""
    want = (doc.get("timing") or {}).get("kind") == "rank_chain"
    got = meta.get("timing") or {}
    if not want:
        if got.get("kind") == "rank_chain":
            return ["meta.timing.kind is rank_chain but family %r is not timed that way" % doc["family"]]
        return []
    errors = []
    if got.get("kind") != "rank_chain":
        errors.append("family %r is timed by rank_chain (shared/rank_timing.py) but meta.timing is %r: "
                      "the leg would time call(), i.e. the host-side rank harness" % (doc["family"], got or None))
    world = ((doc.get("task") or {}).get("const_axes") or {}).get("world")
    if world is not None and got.get("kind") == "rank_chain" and int(got.get("world", 8)) != int(world):
        errors.append("meta.timing.world %r != the task's world axis %r" % (got.get("world", 8), world))
    if not os.path.exists(os.path.join(task_dir, "rank_timing.py")):
        errors.append("rank_timing.py missing from the task (scaffold_ut.py new copies it)")
    try:
        with open(os.path.join(task_dir, "cases.py")) as f:
            src = f.read()
    except OSError:
        src = ""
    if "def rank_chain(" not in src:
        errors.append("cases.py defines no rank_chain(rank, world, group, case)")
    elif "NotImplementedError(\"FILL _Chain" in src:
        errors.append("cases.py rank_chain is still the scaffold's FILL stub")
    return errors


def _lint_leg_env(task_dir, meta):
    """meta.leg_env names "<task file>:<function>" (shared/leg_runner.apply_leg_env); text checks
    only. Every process_state_allow entry names a prefix and says why."""
    errors = []
    decl = meta.get("leg_env")
    if decl:
        fname, sep, fn = str(decl).partition(":")
        fname = fname[:-3] if fname.endswith(".py") else fname
        path = os.path.join(task_dir, fname + ".py")
        try:
            with open(path) as f:
                src = f.read()
        except OSError:
            src = None
        if not sep or not fname or not fn or os.sep in fname:
            errors.append("meta.leg_env %r: expected '<task file>:<function>'" % (decl,))
        elif src is None:
            errors.append("meta.leg_env %r: %s.py is not in the task" % (decl, fname))
        elif not re.search(r"^def %s\(" % re.escape(fn), src, re.M):
            errors.append("meta.leg_env %r: %s.py defines no top-level %s()" % (decl, fname, fn))
    for d in (meta.get("ut_contract") or {}).get("process_state_allow") or []:
        if not (isinstance(d, dict) and d.get("prefix") and (d.get("why") or "").strip()):
            errors.append("process_state_allow entries need prefix and why: %r" % (d,))
    return errors


def package_bind_dir(meta):
    """`kernel_src/<package>` when meta.candidate_bind binds a whole package (a module bind whose
    file is the package's `__init__.py`: overlay_setup copies the whole directory into the
    candidate overlay), else None."""
    b = meta.get("candidate_bind") or {}
    f = str(b.get("file") or "")
    if (b.get("kind") or "module") != "module" or os.path.basename(f) != "__init__.py":
        return None
    return os.path.dirname(os.path.normpath(f))


def tree_file_hashes(root):
    """{path relative to root: sha256 of its bytes} over every file under root, bytecode
    (`__pycache__/`, .pyc, .pyo) excluded."""
    out = {}
    for d, dirs, files in os.walk(root):
        dirs[:] = sorted(x for x in dirs if x != "__pycache__")
        for f in sorted(files):
            if f.endswith((".pyc", ".pyo")):
                continue
            path = os.path.join(d, f)
            with open(path, "rb") as fh:
                out[os.path.relpath(path, root).replace(os.sep, "/")] = \
                    hashlib.sha256(fh.read()).hexdigest()
    return out


BASELINE_PACKAGE = os.path.join("run", "baseline_package.json")


def baseline_package_diff(task_dir, meta):
    """A whole-package bind puts every file of `kernel_src/<package>/` in the candidate overlay,
    so the candidate is whatever differs there from the INSTALLED package (the baseline leg),
    not only what editable_leaves names. `run/baseline_package.json` records the installed
    package ({package, root, files: {path: sha256}}, `scaffold_ut.py baseline-package`, run where
    the package is installed); the binding gate re-checks it against the live install.

    Returns (pkg_dir or None, {"kernel_src/<package>/<path>": "changed"|"added"|"removed"},
    [errors])."""
    pkg = package_bind_dir(meta)
    if pkg is None:
        return None, {}, []
    try:
        with open(os.path.join(task_dir, BASELINE_PACKAGE)) as f:
            rec = json.load(f)
        base = rec["files"]
        if not isinstance(base, dict) or not base:
            raise ValueError("files is empty")
    except (OSError, ValueError, KeyError, TypeError) as e:
        why = e.strerror if isinstance(e, OSError) and e.strerror else e
        return pkg, {}, ["%s (%s): a whole-package bind needs the installed package's file hashes "
                         "to tell what the candidate changes; record them with `scaffold_ut.py "
                         "baseline-package --task <task>` where the package is installed"
                         % (BASELINE_PACKAGE, why)]
    mine = tree_file_hashes(os.path.join(task_dir, pkg))
    diff = {}
    for rel in sorted(set(base) | set(mine)):
        kind = ("added" if rel not in base else "removed" if rel not in mine
                else "changed" if base[rel] != mine[rel] else None)
        if kind:
            diff[pkg + "/" + rel] = kind
    return pkg, diff, []


def check_baseline_package(task_dir, installed_root):
    """The binding gate's half of baseline_package_diff: run/baseline_package.json must still be
    the package the baseline leg imports (`installed_root`), or the lint judged the candidate
    against a stale baseline. Returns [errors]."""
    try:
        with open(os.path.join(task_dir, BASELINE_PACKAGE)) as f:
            recorded = json.load(f)["files"]
    except (OSError, ValueError, KeyError, TypeError) as e:
        return ["%s: %s" % (BASELINE_PACKAGE, e)]
    live = tree_file_hashes(installed_root)
    bad = sorted(p for p in set(recorded) | set(live) if recorded.get(p) != live.get(p))
    if bad:
        return ["%s does not match the installed package at %s in %d file(s) (%s): re-record it "
                "with `scaffold_ut.py baseline-package`" % (BASELINE_PACKAGE, installed_root,
                                                            len(bad), ", ".join(bad[:5]))]
    return []


def _lint_package_diff(task_dir, meta, leaves):
    """Every file of a whole-package bind that differs from the installed package is an editable
    leaf, or is named in `meta.ut_contract.frozen_differs: [{path, why}]` -- a file (or a
    directory, `path` ending in "/") the candidate tree carries changed but this task does not
    claim (another model's subpackage of a shared plugin). Without this a candidate could change
    any file of the package unseen, and the leaves/controls/waivers would cover only what the
    author chose to declare."""
    pkg, diff, errors = baseline_package_diff(task_dir, meta)
    raw = (meta.get("ut_contract") or {}).get("frozen_differs") or []
    if pkg is None:
        if raw:
            errors.append("meta.ut_contract.frozen_differs is only for a whole-package bind "
                          "(candidate_bind of a package __init__.py)")
        return errors
    if not isinstance(raw, list):
        return errors + ["meta.ut_contract.frozen_differs must be a list of {path, why}"]
    leaves = set(leaves)
    files, dirs, seen = set(), [], set()
    for e in raw:
        if not (isinstance(e, dict) and e.get("path") and str(e.get("why") or "").strip()):
            errors.append("frozen_differs entries need path and why: %r" % (e,))
            continue
        path = str(e["path"])
        if path in seen:
            errors.append("frozen_differs names %s twice" % path)
            continue
        seen.add(path)
        if not path.startswith(pkg + "/"):
            errors.append("frozen_differs names %s, which is not under the bound package %s/"
                          % (path, pkg))
        elif path.endswith("/"):
            inside = sorted(l for l in leaves if l.startswith(path))
            if inside:
                errors.append("frozen_differs directory %s holds editable leaves %s: list its "
                              "other files one by one" % (path, inside[:4]))
            elif not any(p.startswith(path) for p in diff):
                errors.append("frozen_differs names %s, where nothing differs from the installed "
                              "package: drop it" % path)
            else:
                dirs.append(path)
        elif path in leaves:
            errors.append("frozen_differs names %s, which is an editable leaf" % path)
        elif path not in diff:
            errors.append("frozen_differs names %s, which does not differ from the installed "
                          "package: drop it" % path)
        else:
            files.add(path)
    for path, kind in sorted(diff.items()):
        if path in leaves and kind != "removed":
            continue
        if path in files or any(path.startswith(d) for d in dirs):
            continue
        errors.append("%s is %s relative to the installed package but is neither an editable leaf "
                      "nor in frozen_differs: the candidate overlay carries it" % (path, kind))
    return errors


def _leaf_control_waivers(mc, leaves, controls, meta=None):
    """`meta.ut_contract.leaf_control_waivers: [{leaf, why}]`: editable leaves that carry no
    negative control of their own. A whole-package candidate (a monokernel bound at its package,
    DESIGN 6.0) edits helpers -- layouts, configs, the device runtime -- that no single numerical
    mutation isolates; each such leaf is named with the reason, never left out silently. Only a
    whole-package bind may waive, and only leaves inside that package.
    Returns ({waived leaf}, [errors])."""
    raw = mc.get("leaf_control_waivers") or []
    if not isinstance(raw, list):
        return set(), ["meta.ut_contract.leaf_control_waivers must be a list of {leaf, why}"]
    pkg = package_bind_dir(meta or {})
    if raw and pkg is None:
        return set(), ["meta.ut_contract.leaf_control_waivers is only for a whole-package bind "
                       "(candidate_bind of a package __init__.py); every leaf of this task needs "
                       "its own negative control"]
    waived, errors = set(), []
    controlled = {c.get("leaf") for c in controls}
    for w in raw:
        if not (isinstance(w, dict) and w.get("leaf") and str(w.get("why") or "").strip()):
            errors.append("leaf_control_waivers entries need leaf and why: %r" % (w,))
            continue
        leaf = w["leaf"]
        if leaf not in leaves:
            errors.append("leaf_control_waivers names %s, which is not an editable leaf" % leaf)
        elif not leaf.startswith(pkg + "/"):
            errors.append("leaf_control_waivers names %s, which is outside the bound package %s/"
                          % (leaf, pkg))
        elif leaf in controlled:
            errors.append("leaf_control_waivers names %s, which has a negative control: drop the "
                          "waiver" % leaf)
        elif leaf in waived:
            errors.append("leaf_control_waivers names %s twice" % leaf)
        else:
            waived.add(leaf)
    # Only a waiver can leave every leaf without a control; with none, each leaf already
    # reports "has no negative control".
    if waived and leaves and not controlled & set(leaves):
        errors.append("every editable leaf is waived: at least one needs a negative control")
    return waived, errors


def lint_task(task_dir):
    """Check a task directory against its frozen contract without importing torch.

    Returns `{"ok": bool, "errors": [...], "warnings": [...], "cases": n}`. Errors block GPU
    gates; warnings are reported. Every rule here is a defect seen in a delivered UT."""
    errors, warnings = [], []
    try:
        doc = load_task_contract(task_dir)
    except (OSError, ValueError) as e:
        return {"ok": False, "errors": ["run/ut_contract.json: %s" % e], "warnings": [], "cases": 0}
    try:
        with open(os.path.join(task_dir, "meta.json")) as f:
            meta = json.load(f)
    except (OSError, ValueError) as e:
        return {"ok": False, "errors": ["meta.json: %s" % e], "warnings": [], "cases": 0}
    try:
        with open(os.path.join(task_dir, "run", "ut_cases.json")) as f:
            manifest = json.load(f)
    except (OSError, ValueError) as e:
        return {"ok": False, "errors": ["run/ut_cases.json: %s" % e], "warnings": [], "cases": 0}

    mc = meta.get("ut_contract") or {}
    if mc.get("family") != doc["family"]:
        errors.append("meta.ut_contract.family %r != frozen contract %r"
                      % (mc.get("family"), doc["family"]))
    if mc.get("contract_sha256") != contract_sha256(doc):
        errors.append("meta.ut_contract.contract_sha256 does not match run/ut_contract.json; "
                      "the contract changed after scaffolding — re-freeze it, do not hand-edit")
    if meta.get("op_kind") and meta["op_kind"] != doc["op_kind"]:
        errors.append("meta.op_kind %r != contract op_kind %r" % (meta["op_kind"], doc["op_kind"]))
    errors.extend(_lint_rank_chain(task_dir, doc, meta))
    errors.extend(_lint_leg_env(task_dir, meta))

    cases = manifest.get("cases") or []
    if not cases:
        errors.append("run/ut_cases.json has no cases")
    sigs = [c.get("sig") for c in cases]
    dup = sorted({s for s in sigs if sigs.count(s) > 1})
    if dup:
        errors.append("duplicate sigs: %s" % dup)
    phase_regimes = meta.get("phase_regimes")
    phases_with_mass, timed_phases = set(), set()
    for c in cases:
        where = "case %s" % c.get("sig")
        if not c.get("sig"):
            errors.append("a case has no sig")
            continue
        try:
            symbols_for(doc, c.get("axes") or {})
        except ContractError as e:
            errors.append("%s: %s" % (where, e))
        regime = c.get("regime")
        if not regime:
            errors.append("%s: no regime (phase)" % where)
        elif isinstance(phase_regimes, dict) and regime not in phase_regimes:
            errors.append("%s: regime %r not in meta.phase_regimes" % (where, regime))
        if c.get("timed"):
            calls = c.get("effective_calls")
            if not isinstance(calls, (int, float)) or isinstance(calls, bool) \
                    or not math.isfinite(calls) or calls < 0:
                errors.append("%s: timed case needs finite effective_calls >= 0, got %r"
                              % (where, calls))
            elif calls > 0:
                phases_with_mass.add(regime)
            timed_phases.add(regime)
    for p in sorted(timed_phases - phases_with_mass):
        errors.append("phase %s has timed cases but no positive effective_calls" % p)
    if not any(c.get("random") for c in cases):
        errors.append("no case has random=true; the value-parity gate would test nothing")

    # A phase that deploys under a CUDA graph needs a replay set spanning >= 2 shapes.
    graph_phases = []
    if isinstance(phase_regimes, dict):
        graph_phases = [p for p, r in phase_regimes.items() if _graph_on(r)]
    elif _graph_on(meta.get("regime") or {}):
        graph_phases = sorted({c.get("regime") for c in cases if c.get("regime")})
    replay = set(manifest.get("replay_sigs") or [])
    for p in graph_phases:
        n = len({c["sig"] for c in cases if c.get("regime") == p and c.get("sig") in replay})
        if n < 2:
            errors.append("phase %s deploys under a CUDA graph but replay_sigs holds %d of its "
                          "cases; need >= 2 distinct boundary shapes" % (p, n))

    wl = meta.get("workload") or {}
    if wl.get("kind") != "agentx_trace_replay" or wl.get("weighting") != "observed":
        warnings.append("meta.workload should be kind=agentx_trace_replay, weighting=observed "
                        "(SKILL.md §4)")
    # The timing gate refuses an AgentX task whose weight model is not an observed source
    # (harness_lib._observed_workload_cases). Without this check a lane passes every other
    # gate and only learns it at the wave's timing tail, hours later.
    swm = wl.get("serving_weight_model") or {}
    if wl.get("kind") == "agentx_trace_replay":
        if swm.get("source") not in OBSERVED_WEIGHT_SOURCES:
            errors.append("meta.workload.serving_weight_model.source is %r; the timing gate accepts "
                          "only %s" % (swm.get("source"), sorted(OBSERVED_WEIGHT_SOURCES)))
        ep = str(swm.get("evidence_path") or "")
        if not ep or ep.startswith("FILL"):
            errors.append("meta.workload.serving_weight_model.evidence_path is unset; point it at "
                          "the observed workload evidence (evidence/observed_workload.json)")
    # harness_lib.serving_weighted_speedup reads meta.workload.cases; they must be the manifest's
    # timed cases, or the weighting and the timed set silently disagree.
    want = {c["sig"]: (str(c.get("regime", "")).lower(), c.get("effective_calls"), c.get("m"))
            for c in cases if c.get("timed") and c.get("sig")}
    have = {c.get("sig"): (str(c.get("regime", "")).lower(), c.get("effective_calls"), c.get("m"))
            for c in (wl.get("cases") or []) if isinstance(c, dict)}
    if want != have:
        diff = sorted(set(want) ^ set(have)) or sorted(s for s in want if want[s] != have.get(s))
        errors.append("meta.workload.cases differ from the manifest's timed cases at %s; "
                      "regenerate them with scaffold_ut.py sync-meta" % diff[:8])

    controls = mc.get("negative_controls") or []
    leaves = mc.get("editable_leaves") or []
    if not leaves:
        errors.append("meta.ut_contract.editable_leaves is empty: name every kernel_src file "
                      "this task claims to deliver")
    waived, waiver_errors = _leaf_control_waivers(mc, leaves, controls, meta)
    errors.extend(waiver_errors)
    errors.extend(_lint_package_diff(task_dir, meta, leaves))
    for leaf in leaves:
        path = os.path.join(task_dir, leaf)
        if not os.path.exists(path):
            errors.append("editable leaf %s does not exist" % leaf)
            continue
        mine = [c for c in controls if c.get("leaf") == leaf]
        if not mine and leaf not in waived:
            errors.append("editable leaf %s has no negative control" % leaf)
        with open(path, errors="replace") as f:
            text = f.read()
        for c in mine:
            n = text.count(c.get("find") or "\0")
            if n != 1:
                errors.append("negative control on %s: `find` occurs %d times, need exactly 1"
                              % (leaf, n))
            if not (c.get("expect") or "").strip():
                errors.append("negative control on %s states no expected deviation" % leaf)
            probes = c.get("probe_sigs", [])
            known = {case["sig"] for case in cases if case.get("random")}
            if (not isinstance(probes, list) or any(not isinstance(s, str) for s in probes)
                    or len(set(probes)) != len(probes) or set(probes) - known):
                errors.append("negative control on %s: probe_sigs must name unique random cases" % leaf)

    # The server compiles the kernel after its framework has been imported; a framework can
    # rebind library functions at import (sglang: triton.next_power_of_2), so a leg that never
    # imports it compiles different code.
    top = str(meta.get("target_callable") or "").split(".")[0]
    if top in SERVING_FRAMEWORKS and not meta.get("server_preimports"):
        errors.append("meta.server_preimports is empty but the entry lives in %s: list the modules "
                      "the server imports before the kernel compiles%s"
                      % (top, " (%s)" % SERVING_FRAMEWORKS[top] if SERVING_FRAMEWORKS[top] else ""))
    ev = meta.get("server_evidence") or {}
    for key in ("compile_variants", "server_calls"):
        if ev.get(key) and not os.path.exists(os.path.join(task_dir, ev[key])):
            errors.append("meta.server_evidence.%s points at a missing file %s" % (key, ev[key]))
    if ev and not os.path.exists(os.path.join(task_dir, "compile_watch.py")):
        errors.append("meta.server_evidence is set but compile_watch.py is not in the task")
    for w in mc.get("compile_variant_waivers") or []:
        if not (w.get("id") and (w.get("why") or "").strip()):
            errors.append("compile_variant_waivers entries need id and why: %r" % (w,))

    for o in (doc.get("task") or {}).get("overrides", []):
        warnings.append("override %s: %s" % (o["path"], o["why"]))
    for name, s in doc["outputs"].items():
        if s["bound"]["kind"] == "harness":
            warnings.append("output %s uses the legacy harness tol; prefer a named bound" % name)
    for name, s in doc["inputs"].items():
        if (s.get("dist") or {}).get("kind") == "routing" and \
                (s["dist"].get("load") or "uniform") == "uniform":
            warnings.append("input %s: uniform routing is the slow end of MoE expert occupancy; "
                            "timing it is a bound, not the served number" % name)
    return {"ok": not errors, "errors": errors, "warnings": warnings, "cases": len(cases)}


def _graph_on(regime):
    """harness_lib.deployment_graph_mode, without importing it: graph-captured unless the
    regime says enforce_eager or cuda_graph is false (decode is captured by default)."""
    if not isinstance(regime, dict) or regime.get("enforce_eager"):
        return False
    return bool(regime.get("cuda_graph", True))


# ----------------------------------------------------------------------------- distributions
def _torch():
    import torch
    return torch


def torch_dtype(name):
    torch = _torch()
    d = getattr(torch, name, None)
    if not isinstance(d, torch.dtype):
        raise ContractError("dtype %r is not available in this torch" % name)
    return d


def _cast(t, dtype_name):
    """Cast a float32 draw to `dtype_name`, saturating at the format's finite max: a plain
    `.to(float8)` of a value past 448 is NaN on the e4m3fn formats."""
    torch = _torch()
    dt = torch_dtype(dtype_name)
    if dtype_name.startswith("float8"):
        t = t.clamp(-torch.finfo(dt).max, torch.finfo(dt).max)
    return t.to(dt)


def draw(name, spec, sym, gen, device, hooks=None):
    """One input tensor from its declared distribution. Returns (stored, logical); logical is
    None unless the stored value is not the number the reference reads (a hook's dequantized
    operand), in which case build_inputs takes the stored values in float64."""
    torch = _torch()
    dist = spec.get("dist") or {}
    kind = dist["kind"]
    shape = shape_of(spec, sym)
    dname = dtype_of(spec, sym)
    if kind == "hook":
        fn = (hooks or {}).get(dist["hook"])
        if fn is None:
            raise ContractError("input %s needs hook %r; define it in cases.HOOKS" % (name, dist["hook"]))
        return fn(name=name, spec=spec, shape=shape, dtype=dname, sym=sym, gen=gen, device=device)
    if kind == "produced":
        raise ContractError("input %s is produced by %s's hook, which did not return it"
                            % (name, dist.get("by")))
    if kind == "fixture":
        raise ContractError("input %s is a structural fixture; load it in cases.py from %s"
                            % (name, dist.get("path")))
    p = {k: (evaluate(v, sym) if isinstance(v, str) and k not in ("kind", "load", "weights",
                                                                    "pattern", "hook", "path")
             else v) for k, v in dist.items()}
    f32 = dict(device=device, dtype=torch.float32)
    if kind in ("normal", "lecun_normal"):
        std = p.get("std", 1.0) if kind == "normal" else 1.0 / math.sqrt(p["fan_in"])
        t = torch.randn(shape, generator=gen, **f32) * std + p.get("mean", 0.0)
    elif kind == "uniform":
        t = torch.rand(shape, generator=gen, **f32) * (p["hi"] - p["lo"]) + p["lo"]
    elif kind == "log_uniform":
        u = torch.rand(shape, generator=gen, **f32)
        t = torch.exp2(u * (p["log2_hi"] - p["log2_lo"]) + p["log2_lo"])
    elif kind == "constant":
        t = torch.full(shape, float(p["value"]), **f32)
    elif kind == "int_range":
        t = _int_range(shape, int(p["lo"]), int(p["hi"]), bool(p.get("distinct")), gen, device)
    elif kind == "permutation":
        n = shape[-1]
        keys = torch.rand(shape, generator=gen, **f32)
        t = keys.argsort(dim=-1).to(torch.float32) / n      # distinct per row: no ties
    elif kind == "routing":
        return _routing(name, shape, dname, p, gen, device)
    elif kind == "simplex":
        # positive rows summing to `total` (default 1): routing weights, mixture coefficients
        t = torch.softmax(torch.randn(shape, generator=gen, **f32) * p.get("temperature", 1.0), -1)
        t = t * p.get("total", 1.0)
    elif kind == "cu_seqlens":
        t = _cu_seqlens(int(p["batch"]), int(p["total"]), int(p.get("min_len", 1)), gen, device)
    elif kind == "seq_lens":
        t = torch.randint(int(p["lo"]), int(p["hi"]) + 1, shape, generator=gen, device=device)
    elif kind == "page_table":
        t = _page_table(shape, int(p["pool"]), gen, device)
    elif kind == "sparse_indices":
        t = _sparse_indices(shape, int(p["context"]), int(p.get("pad", -1)), gen, device)
    else:
        raise ContractError("distribution %s not drawable" % kind)
    if dname in INTEGER_DTYPES:
        return t.to(torch_dtype(dname)), None
    return _cast(t, dname), None


def _int_range(shape, lo, hi, distinct, gen, device):
    torch = _torch()
    if not distinct:
        return torch.randint(lo, hi + 1, shape, generator=gen, device=device)
    n = hi - lo + 1
    k = shape[-1]
    if k > n:
        raise ContractError("cannot draw %d distinct values from [%d, %d]" % (k, lo, hi))
    keys = torch.rand(tuple(shape[:-1]) + (n,), generator=gen, device=device)
    return keys.topk(k, dim=-1).indices + lo


def _routing(name, shape, dname, p, gen, device):
    """Top-k expert ids (distinct per token) and their weights, under a declared expert load.

    `load`: "uniform" (every expert equally likely; the slow end of occupancy), "zipf:<s>"
    (a skewed load), or "observed" with `counts` (per-expert token counts from the trace).
    Only the ids are drawn here; their weights are a separate `simplex` input."""
    torch = _torch()
    n_tokens, topk = shape
    n_experts = int(p["experts"])
    load = p.get("load") or "uniform"
    if load == "uniform":
        logits = torch.rand((n_tokens, n_experts), generator=gen, device=device)
    else:
        if load.startswith("zipf:"):
            s = float(load.split(":", 1)[1])
            w = torch.tensor([1.0 / (i + 1) ** s for i in range(n_experts)], device=device)
        elif load == "observed":
            w = torch.tensor([float(c) for c in p["counts"]], device=device) + 1e-6
            if w.numel() != n_experts:
                raise ContractError("routing counts has %d experts, axis says %d"
                                    % (w.numel(), n_experts))
        else:
            raise ContractError("routing load %r" % load)
        perm = torch.rand(n_experts, generator=gen, device=device).argsort()
        w = w[perm] / w.sum()
        # Gumbel top-k samples k distinct experts with probability proportional to w.
        u = torch.rand((n_tokens, n_experts), generator=gen, device=device).clamp_min(1e-20)
        logits = w.log()[None, :] - (-u.log()).log()
    ids = logits.topk(topk, dim=-1).indices
    return ids.to(torch_dtype(dname)), None


def _cu_seqlens(batch, total, min_len, gen, device):
    torch = _torch()
    if batch * min_len > total:
        raise ContractError("cu_seqlens: %d x %d > total %d" % (batch, min_len, total))
    spare = total - batch * min_len
    cuts = torch.randint(0, spare + 1, (batch - 1,), generator=gen, device=device).sort().values
    edges = torch.cat([torch.zeros(1, device=device, dtype=torch.long), cuts,
                       torch.full((1,), spare, device=device, dtype=torch.long)])
    lens = edges.diff() + min_len
    return torch.cat([torch.zeros(1, device=device, dtype=torch.long), lens.cumsum(0)])


def _page_table(shape, pool, gen, device):
    """Rows of distinct page ids drawn from a shuffled pool: the scattered layout a long-running
    server has, not the contiguous one a fresh allocator hands out."""
    torch = _torch()
    rows, width = (1, shape[0]) if len(shape) == 1 else shape
    if rows * width > pool:
        raise ContractError("page_table: %d x %d pages > pool %d" % (rows, width, pool))
    perm = torch.rand(pool, generator=gen, device=device).argsort()[: rows * width]
    return perm.view(*shape)


def _sparse_indices(shape, context, pad, gen, device):
    """Distinct token slots per row, stratified across the context (the access pattern of a
    top-k selector over a much longer context), padded with `pad` when the row is short."""
    torch = _torch()
    k = shape[-1]
    rows = int(math.prod(shape[:-1]))
    if context <= 0:
        return torch.full(shape, pad, device=device, dtype=torch.long)
    take = min(k, context)
    stride = max(1, context // take)
    base = torch.arange(take, device=device) * stride
    jitter = torch.randint(0, stride, (rows, take), generator=gen, device=device)
    slots = (base[None, :] + jitter).clamp(max=context - 1)
    out = torch.full((rows, k), pad, device=device, dtype=torch.long)
    out[:, :take] = slots
    return out.view(*shape)


class WeightCache:
    """Bounded per-process pristine weights; callers receive independent copies.

    Only built-in, read-only distributions without layout hooks are eligible. Named
    weight RNG streams make cache hits/misses identical and independent of the case's
    activation shape. The protocol is explicitly frozen in new task contracts.
    """
    def __init__(self, max_bytes=2 << 30):
        self.max_bytes = int(max_bytes)
        if self.max_bytes < 0:
            raise ValueError("weight cache budget must be nonnegative")
        self.entries = OrderedDict()
        self.bytes = self.hits = self.misses = 0

    def clear(self):
        self.entries.clear()
        self.bytes = 0

    def draw(self, name, spec, sym, gen, device):
        params = {k: evaluate(v, sym) if isinstance(v, str) and k != "kind" else v
                  for k, v in spec["dist"].items()}
        identity = {"protocol": "weight_substreams/v1", "name": name,
                    "shape": shape_of(spec, sym), "dtype": dtype_of(spec, sym),
                    "distribution": params, "seed": int(gen.initial_seed())}
        digest = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
        # Device index is storage identity, not RNG identity: paired legs on different
        # logical devices must still generate the same seeded numerical inputs.
        torch = _torch()
        dev = torch.device(device)
        if dev.type == "cuda" and dev.index is None:
            dev = torch.device("cuda", torch.cuda.current_device())
        key = (digest, str(dev))
        if key in self.entries:
            self.hits += 1
            value = self.entries.pop(key)
            self.entries[key] = value
            return value.clone(), None
        self.misses += 1
        stream = torch.Generator(device=dev).manual_seed(int(digest[:16], 16) % (2**63))
        value, _ = draw(name, spec, sym, stream, dev)
        size = value.numel() * value.element_size()
        if size <= self.max_bytes and self.max_bytes:
            while self.entries and self.bytes + size > self.max_bytes:
                _, old = self.entries.popitem(last=False)
                self.bytes -= old.numel() * old.element_size()
            self.entries[key] = value
            self.bytes += size
            # Never expose the cached pristine buffer to hooks or device kernels.
            return value.clone(), None
        return value, None


def cacheable_weight(spec, sym=None):
    layout = spec.get("layout")
    layout_active = layout and (sym is None or layout.get("when") is None or evaluate(layout["when"], sym))
    return (spec.get("lifetime") == "weight" and not spec.get("writable")
            and not layout_active and spec.get("dist", {}).get("kind") in
            ("normal", "lecun_normal", "uniform", "log_uniform", "int_range", "constant"))


def build_inputs(doc, case_axes, gen, device, hooks=None, skip=(), with_logical=True, weight_cache=None):
    """All present inputs of one case, in contract order: (stored args, logical args, symbols).

    Inputs named in `skip` are left to the caller (structural fixtures). A `hook` distribution
    may return `{input_name: (stored, logical)}` to produce several inputs from one draw (a
    quantizer's packed values and their scales); those are not drawn again. An input with a
    `layout` `{"hook", "when"}` is drawn row-major, kept row-major as the logical operand, and
    stored through the hook (a preshuffle). Construction order is the contract's, so the same
    seed yields the same operands in every process. Timing legs pass `with_logical=False`: the
    float64 copies are only for the reference, and building them costs memory and time; the
    random draws are identical either way."""
    sym = symbols_for(doc, case_axes)
    args, logical = {}, {}
    for name, spec in doc["inputs"].items():
        if name in skip or name in args or not present(spec, sym):
            continue
        if spec.get("kind") == "scalar":
            args[name] = sym[name] = evaluate(spec.get("value", spec.get("default")), sym)
            continue
        got = (weight_cache.draw(name, spec, sym, gen, device)
               if weight_cache is not None and cacheable_weight(spec, sym)
               else draw(name, spec, sym, gen, device, hooks))
        produced = got if isinstance(got, dict) else {name: got}
        for n, (stored, log) in produced.items():
            if log is None and with_logical:
                log = _f64(stored)
            layout = (doc["inputs"].get(n) or {}).get("layout")
            if layout and (layout.get("when") is None or evaluate(layout["when"], sym)):
                fn = (hooks or {}).get(layout["hook"])
                if fn is None:
                    raise ContractError("input %s needs layout hook %r in cases.HOOKS"
                                        % (n, layout["hook"]))
                stored = fn(stored, name=n, sym=sym)
            args[n] = stored
            if with_logical:
                logical[n] = log
    return args, logical, sym


# ----------------------------------------------------------------------------- input checks
def tensor_digest(obj):
    """sha256 over the bytes of every tensor in `obj` (nested dict/list/tuple), in order."""
    torch = _torch()
    h = hashlib.sha256()

    def walk(o):
        if isinstance(o, torch.Tensor):
            t = o.detach().contiguous().cpu()
            h.update(str((tuple(t.shape), str(t.dtype))).encode())
            h.update(t.view(torch.uint8).numpy().tobytes() if t.numel() else b"")
        elif isinstance(o, dict):
            for k in sorted(o, key=str):
                if not str(k).startswith("_"):
                    h.update(str(k).encode())
                    walk(o[k])
        elif isinstance(o, (list, tuple)):
            for x in o:
                walk(x)
        elif o is None or isinstance(o, (int, float, str, bool)):
            h.update(repr(o).encode())
    walk(obj)
    return h.hexdigest()


def input_snapshot(obj, device_budget=256 << 20):
    """Exact private snapshot, preferring device copies within a bounded GPU budget."""
    torch = _torch()
    left = [device_budget]
    def copy(value):
        if isinstance(value, torch.Tensor):
            size = value.numel() * value.element_size()
            if value.device.type != "cpu" and size > left[0]:
                return value.detach().to("cpu", copy=True)
            left[0] -= size if value.device.type != "cpu" else 0
            return value.detach().clone()
        if isinstance(value, dict):
            return {k: copy(v) for k, v in value.items() if not str(k).startswith("_")}
        if isinstance(value, (list, tuple)):
            return type(value)(copy(v) for v in value)
        return value
    return copy(obj)


def inputs_equal(a, b):
    """Bit-exact comparison; no bulk CPU hashes/copies on the common device path."""
    torch = _torch()
    if isinstance(a, torch.Tensor):
        if not isinstance(b, torch.Tensor) or a.shape != b.shape or a.dtype != b.dtype:
            return False
        x = a.detach().contiguous().reshape(-1).view(torch.uint8)
        y = b.detach().contiguous().reshape(-1).view(torch.uint8)
        if x.device == y.device:
            return torch.equal(x, y)
        # CPU fallback snapshots compare in bounded transfer chunks.
        step = 4 << 20
        return all(torch.equal(x[i:i+step], y[i:i+step].to(x.device))
                   for i in range(0, x.numel(), step))
    if isinstance(a, dict):
        if not isinstance(b, dict):
            return False
        keys = {k for k in b if not str(k).startswith("_")}
        return keys == set(a) and all(inputs_equal(v, b[k]) for k, v in a.items())
    if isinstance(a, (list, tuple)):
        return type(a) is type(b) and len(a) == len(b) and all(inputs_equal(x, y) for x, y in zip(a, b))
    if a is None or isinstance(a, (int, float, str, bool)):
        return type(a) is type(b) and repr(a) == repr(b)
    # Non-tensor objects were outside tensor_digest's comparison; introducing a
    # tensor/container/scalar in their place still changes the checked input set.
    return not (b is None or isinstance(b, (torch.Tensor, dict, list, tuple, int, float, str, bool)))


def check_reproducible(make_inputs, seed, device, *, return_inputs=False):
    """Build one case twice from the same seed; the digests must match. A builder that calls a
    nondeterministic routine (an atomics-based sort, a parallel top-k) fails here, where the
    cause is visible, instead of as 'kernel nondeterminism' inside the correctness gate."""
    torch = _torch()
    a = make_inputs(torch.Generator(device=device).manual_seed(seed))
    snapshot = input_snapshot(a)
    del a
    b = make_inputs(torch.Generator(device=device).manual_seed(seed))
    same = inputs_equal(snapshot, b)
    details = {} if same else {"digest_a": tensor_digest(snapshot), "digest_b": tensor_digest(b)}
    return (same, details, b) if return_inputs else (same, details)


def check_layout_facts(doc, args, sym):
    """Each `layout_facts` item `{"tensor", "attr", "equals", "when"?}` holds on the built args.

    This is what keeps a UT from dropping a layout marker the dispatcher reads (a preshuffled
    weight without `is_shuffled`) and silently timing a different kernel than the one served."""
    bad = []
    for f in doc.get("layout_facts", []):
        if f.get("when") is not None and not evaluate(f["when"], sym):
            continue
        t = args.get(f["tensor"])
        if t is None:
            bad.append("%s absent" % f["tensor"])
            continue
        got = getattr(t, f["attr"], None)
        want = f["equals"]
        if isinstance(want, str) and want.startswith("expr:"):
            want = evaluate(want[5:], sym)
        if got != want:
            bad.append("%s.%s is %r, contract wants %r" % (f["tensor"], f["attr"], got, want))
    return not bad, bad


def check_input_values(doc, args, sym):
    """Integer inputs lie in their declared range (and are distinct where declared); floating
    inputs are finite. Structural operands outside their range make every downstream gate moot."""
    torch = _torch()
    bad = []
    for name, spec in doc["inputs"].items():
        t = args.get(name)
        if not isinstance(t, torch.Tensor) or spec.get("kind") == "scalar":
            continue
        dist = spec.get("dist") or {}
        if t.dtype.is_floating_point and t.element_size() >= 2:
            if not bool(torch.isfinite(t).all()):
                bad.append("%s has non-finite values" % name)
        if dist.get("kind") == "int_range":
            lo, hi = int(evaluate(dist["lo"], sym)), int(evaluate(dist["hi"], sym))
            if t.numel() and (int(t.min()) < lo or int(t.max()) > hi):
                bad.append("%s outside [%d, %d]" % (name, lo, hi))
            if dist.get("distinct") and t.numel():
                s = t.reshape(-1, t.shape[-1]).sort(dim=-1).values
                if bool((s[:, 1:] == s[:, :-1]).any()):
                    bad.append("%s repeats a value within a row" % name)
    return not bad, bad


def writable_inputs(doc):
    """Inputs the live entry writes in place (an in-place rotary, a KV cache it fills). `call()`
    must clone them before invoking the entry, so `inputs_unmodified` still covers them: a
    call() that forgot the clone corrupts the operands of every later draw and replay."""
    return tuple(n for n, s in doc["inputs"].items() if s.get("writable"))


def inputs_unmodified(call, args, writable=()):
    """Run `call(args)` once and confirm no input other than `writable` changed. An extracted
    kernel that writes an operand in place (a residual `add_`, a cache it should only read) makes
    every later draw of the same buffers wrong, and a graph replay doubly so."""
    before = input_snapshot({k: v for k, v in args.items() if k not in writable})
    out = call(args)
    changed = sorted(k for k in before if k not in args or not inputs_equal(before[k], args[k]))
    return not changed, changed, out


# ----------------------------------------------------------------------------- compare
def name_outputs(doc, out, sym=None):
    """Map a call's return (tensor, tuple in contract order, or dict) to the names of the
    outputs present for this case (an absent optional output takes no tuple slot)."""
    names = [n for n, s in doc["outputs"].items() if sym is None or present(s, sym)]
    if isinstance(out, dict):
        return {n: out[n] for n in names if n in out}
    if isinstance(out, (tuple, list)):
        if len(out) < len(names):
            raise ContractError("call returned %d outputs, contract declares %s" % (len(out), names))
        return dict(zip(names, out))
    return {names[0]: out}


def _f64(t):
    torch = _torch()
    if t.dtype in (torch.uint8, torch.int8, torch.int16, torch.int32, torch.int64, torch.bool):
        return t.to(torch.int64)
    if t.element_size() == 1 and not t.dtype.is_floating_point:
        return t.to(torch.int64)
    return t.to(torch.float64)


def compare_one(bound, cand, base, *, ref=None, magnitude=None, base_noise=None, scores=None,
                out_dtype=None, max_violation_fraction=0.0, element_order="fixed", stored=None):
    """One output under one bound. Returns a dict with `ok` and the measures behind it.

    `cand`/`base` are the candidate and frozen-baseline outputs of the same inputs; `ref` the
    float64 reference (or None); `base_noise` a second baseline run of the same inputs, for
    `noise_relative`; `scores` what a `topk_equivalent` selection ranks; `stored` the undifferenced
    compared-against values when cand/base are `delta`s (a tolerance then also admits the
    rounding of storing them, 2u|stored|). Every number is computed on the CPU in float64."""
    torch = _torch()
    kind = bound["kind"]
    cand, base = cand.detach().cpu(), base.detach().cpu()
    res = {"kind": kind, "shape": list(cand.shape)}
    if tuple(cand.shape) != tuple(base.shape):
        res.update(ok=False, why="shape %s != baseline %s" % (tuple(cand.shape), tuple(base.shape)))
        return res
    if kind == "bytes_exact":
        a = cand.contiguous().view(torch.uint8)
        b = base.contiguous().view(torch.uint8)
        diff = int((a != b).sum())
        res.update(ok=diff <= max_violation_fraction * a.numel(), mismatched_bytes=diff, bytes=a.numel())
        return res
    if kind in ("exact", "multiset_rows"):
        a, b = _f64(cand), _f64(base)
        if "value" in bound:          # the output must hold this constant (a stray-write count of 0)
            b = torch.full_like(a, bound["value"])
        if kind == "multiset_rows" or element_order == "any_in_full_rows":
            a, b = a.reshape(-1, a.shape[-1]).sort(dim=-1).values, b.reshape(-1, b.shape[-1]).sort(dim=-1).values
        # NaN on both sides is agreement (as in every other kind); NaN against a number is not.
        bad = int(((a != b) & ~(a.isnan() & b.isnan())).sum())
        res.update(ok=bad <= max_violation_fraction * a.numel(), mismatches=bad, elements=a.numel())
        return res
    if kind == "topk_equivalent":
        return _topk_equivalent(res, bound, cand, base, scores, max_violation_fraction)
    if kind == "harness":
        return _harness_compare(res, cand, base, float(bound["tol"]))

    a, b = cand.to(torch.float64), base.to(torch.float64)
    finite_a, finite_b = torch.isfinite(a), torch.isfinite(b)
    # A non-finite value where the compared-against output is finite is a violation in every
    # kind: a buffer the harness poisoned (NaN) and the kernel never wrote must not pass because
    # `NaN > limit` is False. Where both sides are non-finite they must agree (NaN with NaN).
    nonfinite_mismatch = int(((~finite_b) & ~((a == b) | (a.isnan() & b.isnan()))).sum()
                             + (finite_b & ~finite_a).sum())
    res["nonfinite_mismatches"] = nonfinite_mismatch
    u_out = UNIT_ROUNDOFF.get(out_dtype or "float32", 2.0 ** -24)
    if kind == "sqnr":
        y = ref if ref is not None else b
        keep = torch.isfinite(y)
        if bool((~finite_a[keep]).any()):
            noise, db = math.inf, -math.inf
        else:
            noise = (a - y)[keep].norm().item()
        signal = y[keep].norm().item()
        if noise != math.inf:
            db = math.inf if noise == 0 else (20 * math.log10(signal / noise) if signal else -math.inf)
        res.update(ok=db >= float(bound["min_db"]) and nonfinite_mismatch == 0, sqnr_db=db,
                   min_db=float(bound["min_db"]), against="reference" if ref is not None else "baseline")
        if ref is not None:
            kb = keep & finite_b
            bn = (b - y)[kb].norm().item()
            signal = y[kb].norm().item()
            res["baseline_sqnr_db"] = math.inf if bn == 0 else 20 * math.log10(signal / bn) if signal else -math.inf
        return res

    if kind == "tolerance":
        y = ref if ref is not None else b
        limit = float(bound["atol"]) + (float(bound["rtol"]) + u_out) * y.abs()
        if stored is not None:
            limit = limit + 2 * u_out * stored.detach().cpu().to(torch.float64).abs()
    elif kind == "output_rounding":
        y = ref
        limit = _OUTPUT_ROUNDING_MULTIPLE * u_out * y.abs()
    elif kind == "dot_product":
        if ref is None or magnitude is None:
            raise ContractError("dot_product needs the reference and its magnitude |a|@|b|")
        y = ref
        length = int(bound["length_value"])
        split = int(bound.get("max_split_k", 1))
        n_eff = max(-(-length // s) + (s - 1).bit_length() for s in range(1, split + 1))
        u_acc = UNIT_ROUNDOFF[bound.get("accumulate_dtype", "float32")]
        limit = (_SLACK * (_SQRT_LENGTH_MULTIPLE * math.sqrt(n_eff) + _CONSTANT_MULTIPLE) * u_acc
                 * magnitude.to(torch.float64) + u_out * y.abs())
    elif kind == "baseline_relative":
        if ref is None:
            raise ContractError("baseline_relative needs the float64 reference")
        y = ref
        eb = (b - y).masked_fill(~torch.isfinite(b - y), 0.0)
        rms = eb.pow(2).mean(dim=-1, keepdim=True).sqrt()
        limit = _MAX_ERROR_MULTIPLE * torch.maximum(eb.abs(), rms) + u_out * y.abs()
    elif kind == "noise_relative":
        y = b
        if base_noise is None:
            raise ContractError("noise_relative needs a second baseline run (base_noise)")
        n = (base_noise.to(torch.float64) - b)
        n = n.masked_fill(~torch.isfinite(n), 0.0)
        rms = n.pow(2).mean(dim=-1, keepdim=True).sqrt()
        # `m`/`k` are the README's names for the two multiples; `multiple`/`rounding_multiple`
        # the original ones. Both are read, the short names win.
        multiple = float(bound.get("m", bound.get("multiple", 4.0)))
        floor = float(bound.get("k", bound.get("rounding_multiple", 4.0))) * u_out * b.abs()
        limit = multiple * torch.maximum(n.abs(), rms) + floor
        res["baseline_run_to_run_max"] = float(n.abs().max()) if n.numel() else 0.0
    else:
        raise ContractError("bound %s" % kind)

    keep = torch.isfinite(y) & finite_b
    # Where the compared-against value is finite and the subject's is not, the error is +inf
    # (not NaN, which no comparison flags).
    diff = torch.where(finite_a, a - y, torch.full_like(a, math.inf))
    over = (diff.abs() > limit) & keep
    err = diff[keep]
    lim = limit[keep]
    n = err.numel()
    viol = float((err.abs() > lim).double().mean()) if n else 0.0
    multiples = (err.abs() / lim.clamp_min(1e-300)).masked_fill(err == 0, 0.0)
    peak = float(multiples.max()) if n else 0.0
    res.update(violation_fraction=viol, peak_bound_multiple=peak,
               max_abs_err=float(err.abs().max()) if n else 0.0)
    ok = viol <= max_violation_fraction and nonfinite_mismatch == 0
    if bound.get("per_row") and a.dim():
        # Judge rows (every leading index; the last axis is one row): a row with any element
        # outside the bound is a violating row, and the budget counts rows, not elements.
        bad_rows = over.reshape(-1, a.shape[-1]).any(dim=-1) if a.numel() else over.reshape(0)
        nrows = int(bad_rows.numel())
        res.update(violating_rows=int(bad_rows.sum()), rows=nrows)
        ok = (int(bad_rows.sum()) <= max_violation_fraction * nrows) and nonfinite_mismatch == 0
    if ref is not None and kind in ("dot_product", "output_rounding", "baseline_relative",
                                    "tolerance") and n:
        if not bool(torch.isfinite(err).all()):
            rms, z, unbiased = math.inf, math.inf, False   # an infinite error has no bias test
        else:
            rms = err.pow(2).mean().sqrt().item()
            z = abs(err.mean().item()) / rms if rms > 0 else 0.0
            unbiased = z <= 5 / math.sqrt(n)
        if kind == "tolerance":
            unbiased = unbiased or (viol == 0 and math.isfinite(z))   # no bias test once it holds
        if kind == "baseline_relative" and not unbiased:
            be = (b - y)[keep]
            unbiased = abs(err.mean().item()) <= _MAX_ERROR_MULTIPLE * abs(be.mean().item()) + 5 * rms / n
        res.update(bias_z_score=z, unbiased=unbiased)
        ok = ok and unbiased
    res["ok"] = ok
    return res


def _topk_equivalent(res, bound, cand, base, scores, max_violation_fraction):
    """Index outputs of a top-k: each row's selected scores must be the baseline's multiset of
    selected scores (ties may pick either index), indices must be distinct and in range, and the
    count of valid (non-padding) entries must match. Comparing indices as floats under a relative
    `tol` accepts index 1000 for 1001; comparing them exactly rejects a legal tie order.

    `scores` is the input the selection ranks (`"scores": "<input>"`) or, for a fused score+top-k
    whose scores never leave the kernel, the float64 reference scores (`"scores": "ref:<output>"`,
    with `score_atol` admitting the kernel's own rounding of near-ties)."""
    torch = _torch()
    if scores is None:
        raise ContractError("topk_equivalent: scores %r unavailable" % bound["scores"])
    pad = int(bound.get("pad", -1))
    atol = float(bound.get("score_atol", 0.0))
    rtol = float(bound.get("score_rtol", 0.0))
    s = scores.detach().cpu().to(torch.float64)
    a = cand.to(torch.int64).reshape(-1, cand.shape[-1])
    b = base.to(torch.int64).reshape(-1, base.shape[-1])
    s = s.reshape(a.shape[0], -1)
    offset = int(bound.get("index_offset", 0))
    bad_rows = 0
    reasons = {}
    for r in range(a.shape[0]):
        ar, br = a[r], b[r]
        av, bv = ar[ar != pad] - offset, br[br != pad] - offset
        why = None
        if av.numel() != bv.numel():
            why = "valid_count"
        elif av.numel() and (int(av.min()) < 0 or int(av.max()) >= s.shape[-1]):
            why = "out_of_range"
        elif av.unique().numel() != av.numel():
            why = "duplicate"
        elif not bool(((s[r][av].sort().values - s[r][bv].sort().values).abs()
                       <= atol + rtol * s[r][bv].sort().values.abs()).all()):
            why = "different_scores"
        if why:
            bad_rows += 1
            reasons[why] = reasons.get(why, 0) + 1
    res.update(ok=bad_rows <= max_violation_fraction * a.shape[0], bad_rows=bad_rows,
               rows=a.shape[0], reasons=reasons)
    return res


def _harness_compare(res, cand, base, tol):
    """GEAK's legacy mixed check (`harness_lib.correct`): |c-b| <= tol*RMS(b) + tol*|b|."""
    torch = _torch()
    c, b = cand.float(), base.float()
    fin = torch.isfinite(b) & torch.isfinite(c)
    if bool(((~fin) & (c != b)).any()):
        res.update(ok=False, why="non-finite mismatch")
        return res
    c, b = c[fin], b[fin]
    atol = tol * b.pow(2).mean().sqrt().clamp_min(1e-6) if b.numel() else 0.0
    diff = (c - b).abs()
    ok = bool((diff <= atol + tol * b.abs()).all())
    res.update(ok=ok, max_rel_err=float((diff / (b.abs() + atol)).max()) if b.numel() else 0.0, tol=tol)
    return res


def compare_outputs(doc, cand_out, base_out, *, sym, refs=None, magnitudes=None, base_noise=None,
                    inputs=None, rows=None):
    """Every declared output of one case. `refs`/`magnitudes` hold float64 tensors per output
    (possibly for a row subset `rows` of the leading dimension, in which case cand/base are
    sliced the same way). Returns `{"ok", "outputs": {name: result}}`."""
    torch = _torch()
    cand = name_outputs(doc, cand_out, sym)
    base = name_outputs(doc, base_out, sym)
    noise = name_outputs(doc, base_noise, sym) if base_noise is not None else {}
    cmp = doc["compare"]
    results, ok = {}, True
    for name, spec in doc["outputs"].items():
        if not present(spec, sym):
            continue
        if name not in cand or name not in base:
            results[name] = {"ok": False, "why": "missing from %s" % ("candidate" if name not in cand else "baseline")}
            ok = False
            continue
        c, b, n = cand[name], base[name], noise.get(name)
        if rows is not None:
            c, b = c[rows], b[rows]
            n = n[rows] if n is not None else None
        bound = dict(spec["bound"])
        if "length" in bound:
            bound["length_value"] = int(evaluate(bound["length"], sym))
        out_dtype = dtype_of(spec, sym) if "dtype" in spec else None
        scores = None
        if bound["kind"] == "topk_equivalent":
            src = bound["scores"]
            scores = (refs or {}).get(src[4:]) if src.startswith("ref:") else (inputs or {}).get(src)
            if scores is not None and rows is not None and not src.startswith("ref:"):
                scores = scores[rows]
        if spec.get("sort_by"):
            # rows of this output follow another output's order (weights follow their ids)
            c = c.gather(-1, cand[spec["sort_by"]][rows if rows is not None else slice(None)]
                         .to(torch.int64).argsort(dim=-1))
            b = b.gather(-1, base[spec["sort_by"]][rows if rows is not None else slice(None)]
                         .to(torch.int64).argsort(dim=-1))
        ref = None if bound.get("against") == "baseline" else (refs or {}).get(name)
        produced = _produced(spec, c, b)
        if produced is None and spec.get("require_finite"):
            produced = _finite(c, b, ref)
        stored = None
        if produced is None and bound.get("delta"):
            stored = b if ref is None else ref
            c, b, ref, produced = _delta(spec, bound["delta"], c, b, ref, cand, base, refs, rows)
        if produced is not None:
            results[name] = produced
            ok = ok and bool(produced["ok"])
            continue
        r = compare_one(bound, c, b, ref=ref, magnitude=(magnitudes or {}).get(name), stored=stored,
                        base_noise=n, scores=scores, out_dtype=out_dtype,
                        max_violation_fraction=float(spec.get("max_violation_fraction",
                                                              cmp.get("max_violation_fraction", 0.0))),
                        element_order=spec.get("element_order", cmp.get("element_order", "fixed")))
        results[name] = r
        ok = ok and bool(r.get("ok"))
    return {"ok": ok, "outputs": results}


def _finite(cand, base, ref):
    """`require_finite`: an output the harness poisons with NaN (KV rows, residual stream) must be
    finite on both sides. NaN on both sides at the same place is otherwise agreement, so a
    poisoning or gather mistake common to every leg (all legs share the rank harness) would
    pass the candidate and the oracle floor alike. None: compare normally."""
    torch = _torch()
    sides = [("reference" if ref is not None else "baseline", ref if ref is not None else base),
             ("baseline", base if ref is not None else None), ("subject", cand)]
    for side, t in sides:
        if t is None or not t.dtype.is_floating_point:
            continue
        bad = int((~torch.isfinite(t.detach().cpu())).sum())
        if bad:
            return {"ok": False, "status": "%s_nonfinite" % side, "nonfinite": bad,
                    "elements": int(t.numel()),
                    "why": "the %s holds %d non-finite values (a poisoned slot never written, or a "
                           "harness gather/poison-timing error)" % (side, bad)}
    return None


def _delta(spec, d, c, b, ref, cand, base, refs, rows):
    """The step's change (`bound.delta`) of cand / base / ref: `x - minus`, or with `along` the
    consecutive differences over that axis, the first against `minus`. Returns (c, b, ref,
    failure-or-None); a compared-against change that is identically zero fails (the step did
    nothing, or the harness returned the input as the output)."""
    torch = _torch()
    minus = d["minus"]
    axis = spec["shape"].index(d["along"]) if "along" in d else None

    def diff(x, m):
        x, m = x.detach().cpu().to(torch.float64), m.detach().cpu().to(torch.float64)
        if rows is not None:
            m = m[rows]
        if axis is None:
            return x - m
        prev = torch.cat([m.unsqueeze(axis), x.narrow(axis, 0, x.shape[axis] - 1)], dim=axis)
        return x - prev

    for side, outs in (("candidate", cand), ("baseline", base)):
        if minus not in outs:
            return c, b, ref, {"ok": False, "why": "delta.minus %s missing from the %s" % (minus, side)}
    c2, b2 = diff(c, cand[minus]), diff(b, base[minus])
    r2 = None
    if ref is not None:
        if minus not in (refs or {}):
            raise ContractError("delta.minus %s has no reference to difference against" % minus)
        r2 = diff(ref, refs[minus])
    y = r2 if r2 is not None else b2
    if y.numel() and bool((y[torch.isfinite(y)] == 0).all()):
        return c2, b2, r2, {"ok": False, "status": "reference_delta_zero",
                            "why": "the compared-against side's change (output - %s) is zero" % minus}
    return c2, b2, r2, None


def _produced(spec, cand, base):
    """`optional_if_not_produced: {"poison": v}`: the harness fills the buffer with `v` before
    the call. A subject that left all of it at `v` did not produce the output (an optimized
    kernel may legitimately stop exposing it): reported `not_produced`, neither passed against
    stale values nor failed. A subject that wrote only part of it fails. The compared-against
    side (the oracle / baseline floor) must produce every element: any poison left there fails,
    all of it (`reference_not_produced`) or part (`reference_partially_written`), so a subject
    whose holes match the reference's cannot pass as -7 == -7. None: compare normally."""
    opt = spec.get("optional_if_not_produced")
    if not opt:
        return None
    v = opt["poison"]
    c_hit = cand.detach().cpu() == v
    b_hit = base.detach().cpu() == v
    if b_hit.numel() and bool(b_hit.all()):
        return {"ok": False, "status": "reference_not_produced", "poison": v,
                "why": "the compared-against side left every element at the poison value %r" % v}
    if bool(b_hit.any()):
        return {"ok": False, "status": "reference_partially_written", "poison": v,
                "unwritten": int(b_hit.sum()), "elements": int(b_hit.numel()),
                "why": "the compared-against side left part of the output at the poison value %r" % v}
    if c_hit.numel() and bool(c_hit.all()):
        return {"ok": True, "status": "not_produced", "poison": v}
    if bool(c_hit.any()):
        return {"ok": False, "status": "partially_written", "poison": v,
                "unwritten": int(c_hit.sum()), "elements": int(c_hit.numel()),
                "why": "part of the output still holds the poison value %r" % v}
    return None


def control_margin(result):
    """How far a negative control landed outside the bound: the worst output's bound multiple,
    mismatch count or SQNR deficit. A control 'failing' by 1.01x is not evidence the gate can
    reject anything; `ut_driver` wants >= `min_margin`."""
    worst = 0.0
    for r in result.get("outputs", {}).values():
        if r.get("ok"):
            continue
        if "peak_bound_multiple" in r:
            worst = max(worst, r["peak_bound_multiple"])
        elif "sqnr_db" in r:
            worst = max(worst, 10 ** ((r["min_db"] - r["sqnr_db"]) / 20) if math.isfinite(r["sqnr_db"]) else math.inf)
        else:
            worst = max(worst, math.inf)
    return worst


def main(argv=None):
    import argparse
    ap = argparse.ArgumentParser(description="Lint a task against its frozen UT contract (CPU only).")
    ap.add_argument("task_dir")
    ap.add_argument("--out", help="write the verdict JSON here as well")
    a = ap.parse_args(argv)
    verdict = lint_task(os.path.abspath(a.task_dir))
    text = json.dumps(verdict, indent=2)
    print(text)
    if a.out:
        with open(a.out, "w") as f:
            f.write(text + "\n")
    return 0 if verdict["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
