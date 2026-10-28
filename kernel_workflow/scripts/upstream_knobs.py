#!/usr/bin/env python3
"""List the knobs a candidate kernel's upstream has and the candidate does not.

    python3 kernel_workflow/scripts/upstream_knobs.py <candidate.py|dir> [--upstream DIR]...
    python3 kernel_workflow/scripts/upstream_knobs.py kernel_src/ --json

WHY THIS EXISTS
A task's kernel is often a VENDORED CUT of an upstream one: same function name, same body shape,
fewer parameters. Everything the cut dropped is invisible from inside the file -- there is no import
to follow, no `NotImplementedError`, nothing to grep. The cut usually also converts the parameters
it kept-but-abandoned into assertions, so the file reads as if the axis had been measured and
closed, when it was only never ported.

Measured, on the GLM-5.2 MXFP4 fused-MoE task. Its `fused_moe_seam.py` defines

    def compile_gemm1_a4w4_port(BM=32, use_nt=True, inline_quant=False, *,
                                D_HIDDEN, D_INTER, NE, TOPK, BN=256, BK=256,
                                interleave=False, xcd_swizzle=0):
        ...
        assert BN == 256 and BK == 256, "only BN==BK==256 supported"

while upstream's function OF THE SAME NAME also takes `prefetch_hidden`, `a_dtype`, `out_dtype`,
`act`, `situ_beta`, `situ_linear_beta`, `swiglu_limit`, `enable_bias`, `native_scale_layout`,
`num_waves` and `k_wave` -- and varies `BN`. Three campaign rounds concluded "BN is inert" from
inside the seam; the assert was the whole reason. Re-opening BN (256 -> 128, which doubles
`num_n_blocks` and so the decode grid) was one of the three mechanisms that took that task from
1.1294x to 1.3205x. The other two came from axes written off the same way.

`check_flydsl_compat.py` cannot find this: it validates the symbols a candidate ALREADY USES, so an
unused upstream knob is structurally outside its question. This file asks the complementary one.

WHAT IT REPORTS
  narrowed  a parameter upstream's same-named function has and the candidate's does not
  pinned    a parameter the candidate declares and then asserts to a constant -- a closed axis that
            was closed by the port, not by a measurement
  fuzzy     the same, where the names matched only after stripping a vendoring suffix
            (`_port`, `_fold`, `_clone`, `_v2`, ...), so confirm the pairing before trusting it
  coverage  how many of the modules the candidate imports out of this tree's own packages the tree
            actually contains -- because the wrong upstream revision reports an EMPTY findings list,
            which is indistinguishable from a clean one. See `stack_coverage`.

Static, AST-only, both sides. These modules do not import without a GPU and a matching FlyDSL, and a
tool that needs the stack working cannot run at the point in a campaign where this question is
worth asking. Nothing here is a verdict: a knob upstream has may be absent for a good reason. It is
a list of things to try, and the point is that the list should not have to be assembled by hand.
"""
from __future__ import annotations

import argparse
import ast
import json
import os
import re
import sys
from pathlib import Path

# Suffixes a vendored copy conventionally appends. Checked longest-first so `_a4w4_port` does not
# match `_port` and stop.
VENDOR_SUFFIX = re.compile(r"(?:_port|_fold|_clone|_vendored|_local|_seam|_copy|_v\d+|\d+)$")

# Where to look for the upstream tree when none is given. Same contract as check_flydsl_compat's
# `pin_task_stack`: the task declares its own stack and that declaration wins over the box's.
STACK_FILES = (".geak_pythonpath", ".geak_stack")


def params(fn):
    """Ordered parameter names of a function def, with how each may be passed.

    `self` is dropped: a method's receiver is not a knob, and keeping it reports every vendored
    class method as having narrowed by one.
    """
    a = fn.args
    out = []
    for arg in a.posonlyargs:
        out.append((arg.arg, "positional-only"))
    for arg in a.args:
        out.append((arg.arg, "positional"))
    if a.vararg:
        out.append(("*" + a.vararg.arg, "var-positional"))
    for arg in a.kwonlyargs:
        out.append((arg.arg, "keyword-only"))
    if a.kwarg:
        out.append(("**" + a.kwarg.arg, "var-keyword"))
    return [(n, k) for n, k in out if n != "self"]


def _literal(node):
    try:
        return ast.literal_eval(node)
    except (ValueError, SyntaxError):
        return None


def pinned_params(fn, names):
    """Parameters of `fn` that its own body fixes to a constant.

    Three spellings, all seen in vendored kernels:
        assert BN == 256                     -> pinned
        assert BN in (256, 512)              -> pinned to a set
        if BN != 256: raise AssertionError   -> pinned

    A comparison against another NAME (`assert BN == BK`) is a consistency check, not a pin, and is
    not reported -- it closes nothing on its own.
    """
    found = {}

    def note(name, value):
        if name in names:
            found.setdefault(name, value)

    def from_compare(cmp_node, invert=False):
        if not isinstance(cmp_node, ast.Compare) or len(cmp_node.ops) != 1:
            return
        op, left, right = cmp_node.ops[0], cmp_node.left, cmp_node.comparators[0]
        if not isinstance(left, ast.Name):
            return
        want_eq = (isinstance(op, ast.Eq) and not invert) or (isinstance(op, ast.NotEq) and invert)
        want_in = (isinstance(op, ast.In) and not invert) or (isinstance(op, ast.NotIn) and invert)
        if want_eq or want_in:
            val = _literal(right)
            if val is not None:
                note(left.id, val)

    for node in ast.walk(fn):
        if isinstance(node, ast.Assert):
            test = node.test
            for part in (test.values if isinstance(test, ast.BoolOp) and
                         isinstance(test.op, ast.And) else [test]):
                from_compare(part)
        elif isinstance(node, ast.If) and any(
                isinstance(s, ast.Raise) for s in node.body):
            test = node.test
            for part in (test.values if isinstance(test, ast.BoolOp) and
                         isinstance(test.op, ast.Or) else [test]):
                from_compare(part, invert=True)
    return found


def top_level_defs(path):
    """`{name: FunctionDef}` for the module-level functions of a file, '' on a parse failure.

    Module level only. A nested closure is an implementation detail of its parent, and matching one
    across two trees by bare name pairs things that are not counterparts.
    """
    try:
        tree = ast.parse(Path(path).read_text(encoding="utf-8", errors="replace"))
    except (OSError, SyntaxError) as exc:
        return {}, f"{path}: {exc.__class__.__name__}: {exc}"
    out = {}
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            out.setdefault(node.name, node)
    return out, None


def imported_modules(path):
    """Dotted module names a candidate imports, as a set.

    Used to break same-name ties. Without it `flat_buffer_view` in a fused-MoE seam pairs against a
    `conv3d_gfx950_utils.py` function of the same name -- a real collision in this very tree -- and
    the report's first finding is a coincidence, which is how a reader learns to skim it.
    """
    try:
        tree = ast.parse(Path(path).read_text(encoding="utf-8", errors="replace"))
    except (OSError, SyntaxError):
        return set()
    mods = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            mods.update(a.name for a in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module and not node.level:
            mods.add(node.module)
            mods.update(f"{node.module}.{a.name}" for a in node.names)
    return mods


def _catches_import(handler):
    """Does this `except` clause tolerate a failed import?"""
    exc = handler.type
    if exc is None:  # bare `except:`
        return True
    names = exc.elts if isinstance(exc, ast.Tuple) else [exc]
    return any(getattr(n, "id", getattr(n, "attr", None))
               in {"ImportError", "ModuleNotFoundError", "Exception", "BaseException"}
               for n in names)


def _guarded_imports(tree):
    """`id()` of every import the file itself declares optional.

    A `try: from aiter.ops.flydsl.utils import ... except ImportError:` is the author stating that
    this module may legitimately be absent, usually behind a feature probe. Counting it against
    coverage makes the correct tree look wrong -- on the GLM-5.2 seam it was the single remaining
    false alarm -- and worse, it is the one import whose absence is guaranteed harmless.
    """
    out = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Try) and any(_catches_import(h) for h in node.handlers):
            for stmt in node.body:
                for sub in ast.walk(stmt):
                    if isinstance(sub, (ast.Import, ast.ImportFrom)):
                        out.add(id(sub))
    return out


def import_targets(path):
    """Module names a candidate imports -- true modules only, and only unguarded ones.

    `imported_modules` deliberately also emits `mod.name` for every `from mod import name`, because
    for RANKING a composite that happens to be a real submodule is a stronger signal than the
    package alone. For COVERAGE that same composite is a false alarm: `from aiter import flat_view`
    would report a missing module `aiter.flat_view` that was never a module. So this is the strict
    set, and only this one may conclude anything is absent.
    """
    try:
        tree = ast.parse(Path(path).read_text(encoding="utf-8", errors="replace"))
    except (OSError, SyntaxError):
        return set()
    guarded = _guarded_imports(tree)
    mods = set()
    for node in ast.walk(tree):
        if id(node) in guarded:
            continue
        if isinstance(node, ast.Import):
            mods.update(a.name for a in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module and not node.level:
            mods.add(node.module)
    return mods


def module_in_roots(name, roots):
    """Can `import name` be satisfied by these roots, treated as sys.path entries?

    Resolved against the FILESYSTEM, not against `dotted()`'s index. `dotted()` walks up while
    `__init__.py` exists because that is the question an upstream FILE asks ("what is my importable
    name"), but PEP 420 namespace packages have no `__init__.py` -- `aiter/utility/` and
    `flydsl/_mlir/` on this box are both -- so going through it reported real, present modules as
    missing and the warning fired on the correct tree. Here the question is the importer's own:
    does `<root>/a/b/c.py` or `<root>/a/b/c/` exist. Compiled extensions count: `_mlir_libs` ships
    as `.so`, and a seam importing one is not evidence of a wrong revision.
    """
    rel = Path(*name.split("."))
    for root in roots:
        root = Path(root)
        if (root / rel).is_dir() or (root / rel).with_suffix(".py").is_file():
            return True
        parent = root / rel.parent
        if parent.is_dir() and any(parent.glob(rel.name + ".*.so")):
            return True
    return False


def stack_coverage(cand_files, roots):
    """`(imported_from_upstream, missing)` -- does this tree even contain what the candidate imports?

    Measured, same GLM-5.2 task as the module docstring. Pointed at the box's ambient checkout
    (`/sgl-workspace/aiter`, which predates the whole `mxfp4_*` family the seam imports) this tool
    reports 2242 upstream functions, 2 findings and ZERO pins. Pointed at the stack the task
    actually pins it reports 7664, 86 and 6 -- including the `BN == 256` pin that three rounds had
    concluded was inert. The failure is silent and it looks exactly like a clean result: an empty
    findings list reads as "nothing was closed by the port", not as "you indexed the wrong tree".

    So before trusting an empty report, check the cheap structural thing: of the modules the
    candidate imports out of a package this tree provides, how many does the tree actually have?
    Only packages the tree claims are considered -- a seam importing `torch` or `triton` says
    nothing about whether the aiter root is the right revision.
    """
    want, missing = set(), set()
    for f in cand_files:
        for m in import_targets(f):
            # The top-level package must be one of these roots' own, or the import is about some
            # other dependency entirely and its absence here means nothing.
            if not module_in_roots(m.split(".", 1)[0], roots):
                continue
            want.add(m)
            if not module_in_roots(m, roots):
                missing.add(m)
    return want, missing


def dotted(root, file):
    """Importable module name of an upstream file: `.../aiter/ops/x.py` -> `aiter.ops.x`.

    Found by walking up while `__init__.py` exists, which is what an importer does. Deriving it
    from the passed root instead assumes the root IS the package directory -- true for
    `--upstream .../aiter_main/aiter/aiter` and false for a root holding several packages, where
    every module then carries a leading component that matches nothing the candidate can import
    and the whole import ranking silently reads 0.
    """
    f = Path(file).resolve()
    parts = [f.stem]
    d = f.parent
    while (d / "__init__.py").is_file():
        parts.append(d.name)
        if d.parent == d:
            break
        d = d.parent
    return ".".join(reversed(parts))


def affinity(module, imports):
    """Rank an upstream module against what the candidate imports. Higher is better.

    Ranking only. Do NOT read a low score as "this is a collision": a VENDORED cut does not import
    the file it was cut from -- that is what vendoring means -- so the true counterpart of the
    central case scores no better than a coincidence. Where several modules define the same name
    they are all reported and the reader pairs them; a tool that picked one and hid the rest would
    be most confidently wrong exactly where this is most worth running.
    """
    if not module:
        return (0, 0)
    parts = module.split(".")
    best = 0
    for imp in imports:
        ip = imp.split(".")
        n = 0
        while n < min(len(parts), len(ip)) and parts[n] == ip[n]:
            n += 1
        best = max(best, n)
    return (1 if module in imports else 0, best)


def index_upstream(roots):
    """`{function_name: [(file, FunctionDef)]}` over every .py under the upstream roots."""
    index, errors = {}, []
    for root in roots:
        root = Path(root)
        files = sorted(root.rglob("*.py")) if root.is_dir() else [root]
        for f in files:
            # A build tree's generated copies and a package's own tests are not the upstream of
            # anything; pairing against them produces matches nobody can act on.
            parts = set(f.parts)
            if parts & {"build", "dist", "__pycache__", ".git", "op_tests", "test", "tests"}:
                continue
            defs, err = top_level_defs(f)
            if err:
                errors.append(err)
                continue
            for name, node in defs.items():
                index.setdefault(name, []).append((str(f), node, dotted(root, f)))
    return index, errors


def pinned_stack(start):
    """Upstream roots declared by the task, nearest declaration first; [] if none."""
    start = Path(start).resolve()
    roots = []
    for directory in [start if start.is_dir() else start.parent, *start.parents]:
        for name in STACK_FILES:
            stack = directory / name
            if not stack.is_file():
                continue
            for line in stack.read_text(encoding="utf-8", errors="replace").splitlines():
                entry = line.split("#", 1)[0].strip()
                if entry and Path(entry).is_dir() and entry not in roots:
                    roots.append(entry)
        if roots:
            break  # nearest wins; stacks up the tree are not merged
    return roots


def compare(cand_path, upstream_index):
    """Findings for one candidate file."""
    defs, err = top_level_defs(cand_path)
    if err:
        return [], [err]
    imports = imported_modules(cand_path)
    findings = []
    for name, fn in sorted(defs.items()):
        if name.startswith("_") and name not in upstream_index:
            continue  # a private helper with no upstream counterpart is the file's own
        match_name, how = name, "exact"
        if name not in upstream_index:
            stem = VENDOR_SUFFIX.sub("", name)
            # Strip repeatedly: `compile_gemm1_a4w4_port_v2` -> `compile_gemm1_a4w4_port` -> ...
            while stem != name and stem not in upstream_index:
                nxt = VENDOR_SUFFIX.sub("", stem)
                if nxt == stem:
                    break
                stem = nxt
            if stem == name or stem not in upstream_index:
                continue
            match_name, how = stem, "fuzzy"

        cand = params(fn)
        cand_names = {n for n, _ in cand}
        pins = pinned_params(fn, cand_names)

        # Every upstream file defining that name, best-ranked first. More than one is common and
        # informative -- `compile_gemm2_a4w4_port` exists both as the kernel and as a dispatcher
        # wrapper with 19 further knobs, and which is "the" upstream is the reader's call.
        counterparts = []
        for up_file, up_fn, up_mod in sorted(upstream_index[match_name],
                                             key=lambda t: affinity(t[2], imports), reverse=True):
            up = params(up_fn)
            up_names = {x for x, _ in up}
            missing = [(n, k) for n, k in up
                       if n not in cand_names and not n.startswith(("*", "**"))]
            counterparts.append({
                "file": up_file, "line": up_fn.lineno, "module": up_mod,
                "imported_by_candidate": up_mod in imports,
                "narrowed": [{"param": n, "kind": k} for n, k in missing],
                "pinned_upstream_has": sorted(n for n in pins if n in up_names),
            })
        # Lead with whichever counterpart exposes the most, after the import ranking -- the widest
        # one is the one worth reading first, and the others stay listed underneath.
        counterparts.sort(key=lambda c: (c["imported_by_candidate"], len(c["narrowed"])),
                          reverse=True)
        if not counterparts or (not counterparts[0]["narrowed"] and not pins):
            continue
        primary = counterparts[0]
        findings.append({
            "function": name,
            "upstream_function": match_name,
            "match": how,
            "candidate_file": str(cand_path),
            "candidate_line": fn.lineno,
            "upstream_file": primary["file"],
            "upstream_line": primary["line"],
            "upstream_module": primary["module"],
            "narrowed": primary["narrowed"],
            "pinned": [{"param": n, "to": v,
                        "upstream_has": n in set(primary["pinned_upstream_has"])}
                       for n, v in sorted(pins.items())],
            "other_counterparts": counterparts[1:4],
        })
    return findings, []


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("candidate", nargs="+", help="candidate .py file(s) or directory")
    ap.add_argument("--upstream", action="append", default=[], metavar="DIR",
                    help="upstream tree to compare against; repeatable. Default: the roots the "
                         "task pins in .geak_pythonpath next to the candidate.")
    ap.add_argument("--json", action="store_true")
    ap.add_argument("--fail-on-narrowing", action="store_true",
                    help="exit 1 when anything is found (for CI; off by default -- this is a "
                         "list of things to try, not a verdict)")
    a = ap.parse_args(argv)

    cand_files = []
    for raw in a.candidate:
        p = Path(raw)
        cand_files.extend(sorted(p.rglob("*.py")) if p.is_dir() else [p])
    cand_files = [f for f in cand_files if "__pycache__" not in f.parts]
    if not cand_files:
        print("no candidate .py files", file=sys.stderr)
        return 2

    roots = list(a.upstream) or pinned_stack(cand_files[0])
    if not roots:
        print("no upstream tree: pass --upstream DIR, or drop a .geak_pythonpath next to the "
              "candidate naming the stack it was written against", file=sys.stderr)
        return 2

    index, idx_errors = index_upstream(roots)
    findings, errors = [], list(idx_errors)
    for f in cand_files:
        got, errs = compare(f, index)
        findings.extend(got)
        errors.extend(errs)

    want, missing = stack_coverage(cand_files, roots)
    # Loud, on stderr, in both modes: an empty findings list is only meaningful once the tree is
    # known to be the right one, and the wrong tree produces the same empty list.
    if missing:
        print(f"WARNING: wrong upstream tree? {len(missing)} of the {len(want)} module(s) these "
              f"candidates import from this tree's own packages do not exist in it:",
              file=sys.stderr)
        for m in sorted(missing)[:8]:
            print(f"  missing: {m}", file=sys.stderr)
        if len(missing) > 8:
            print(f"  ... and {len(missing) - 8} more", file=sys.stderr)
        print("  A report with few or no findings is NOT a clean result until this is resolved. "
              "Point --upstream at the revision the kernel was written against.", file=sys.stderr)

    if a.json:
        print(json.dumps({"upstream_roots": roots, "candidates": len(cand_files),
                          "upstream_functions": len(index),
                          "stack_coverage": {"imported_from_upstream": len(want),
                                             "missing": sorted(missing)},
                          "findings": findings,
                          "errors": errors}, indent=2))
    else:
        print(f"upstream: {', '.join(roots)}  ({len(index)} module-level functions, "
              f"{len(want) - len(missing)}/{len(want)} imported module(s) present)")
        print(f"candidate: {len(cand_files)} file(s) -> {len(findings)} function(s) with "
              f"knobs upstream has and this does not\n")
        # Widest narrowing first, and a pinned parameter outranks a merely absent one: a pin is a
        # closed axis the reader probably believes was measured.
        order = sorted(findings, key=lambda f: (-len(f["pinned"]), -len(f["narrowed"])))
        for f in order:
            tag = "" if f["match"] == "exact" else "  [FUZZY NAME MATCH -- confirm the pairing]"
            print(f"{os.path.basename(f['candidate_file'])}:{f['candidate_line']} "
                  f"{f['function']}(){tag}")
            print(f"  upstream {f['upstream_module'] or f['upstream_file']}:"
                  f"{f['upstream_line']} {f['upstream_function']}()")
            if f["narrowed"]:
                print(f"  narrowed ({len(f['narrowed'])}): "
                      + ", ".join(n["param"] for n in f["narrowed"]))
            for p in f["pinned"]:
                where = "upstream varies it" if p["upstream_has"] else "not an upstream parameter"
                print(f"  pinned:   {p['param']} == {p['to']!r}  ({where})")
            for o in f["other_counterparts"]:
                print(f"  also defined in {o['module'] or o['file']}:{o['line']} "
                      f"({len(o['narrowed'])} narrowed)")
            print()
        if errors:
            print(f"({len(errors)} file(s) could not be parsed)", file=sys.stderr)

    return 1 if (findings and a.fail_on_narrowing) else 0


if __name__ == "__main__":
    sys.exit(main())
