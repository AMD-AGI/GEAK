#!/usr/bin/env python3
"""Regenerate version_map.md by statically diffing two or more installed FlyDSL trees.

Why static: two FlyDSL versions cannot be imported into one process, and the recipes that need this
table run on a box that has exactly one of them installed. Everything here is `ast`-level, so the
table can be built for versions the current box cannot execute.

What the table answers: a recipe validated on FlyDSL X says "map the same invariants onto the APIs
available in your version". This resolves the mechanical half of that -- which symbol moved where,
which one is gone, which module disappeared. It deliberately does NOT judge whether a replacement is
semantically equivalent; a symbol that survived as a raw intrinsic after its wrapper was deleted is
reported as `moved`, and the caller still has to re-derive the wrapper's behaviour.

Run from perf_knowledge/:
    python3 languages/flydsl/_gen_version_map.py \
        --root /tmp/fly020_bak --root /tmp/geak-flydsl-0.2.2 \
        --root /tmp/fly024 --root /opt/venv/lib/python3.10/site-packages \
        --check-imports /sgl-workspace/aiter/aiter/ops/flydsl

Each --root is a directory *containing* a `flydsl/` package; the version is read from its
`flydsl/__init__.py`. The cross-version tables are static, so they cover versions this box cannot
run. --check-imports is separate and optional: it resolves call sites against the flydsl INSTALLED
in this interpreter, which finds a stale import before it raises at runtime.
"""
import argparse
import ast
import importlib
import os
import re
import sys
from collections import OrderedDict

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_OUT = os.path.join(HERE, "version_map.md")
NOTES = os.path.join(HERE, "version_map_notes.yaml")


# --------------------------------------------------------------------------------------------------
# Reading one tree
# --------------------------------------------------------------------------------------------------

def read_text(path):
    with open(path, encoding="utf-8", errors="replace") as f:
        return f.read()


def read_version(root):
    """Version string from <root>/flydsl/__init__.py, or None when this is not a FlyDSL tree."""
    init = os.path.join(root, "flydsl", "__init__.py")
    if not os.path.isfile(init):
        return None
    try:
        tree = ast.parse(read_text(init))
    except SyntaxError:
        return None
    for node in tree.body:
        if isinstance(node, ast.Assign):
            for tgt in node.targets:
                if isinstance(tgt, ast.Name) and tgt.id == "__version__":
                    try:
                        return str(ast.literal_eval(node.value))
                    except (ValueError, SyntaxError):
                        return None
    return None


def module_name(pkg_root, path):
    """`flydsl/expr/arith.py` -> `flydsl.expr.arith`; a package __init__ keeps the package name."""
    rel = os.path.relpath(path, os.path.dirname(pkg_root))
    rel = rel[: -len(".py")]
    parts = rel.split(os.sep)
    if parts[-1] == "__init__":
        parts = parts[:-1]
    return ".".join(parts)


def _module_facts(tree):
    """(public def/class names, every module-level bound name, star_import_or_getattr).

    Two different questions need two different name sets. "Where does this symbol live" is answered
    by definitions only -- counting a re-export would report a symbol as living in every module that
    forwards it. "Will `from M import S` resolve" is answered by everything bound at module level,
    including constants, private helpers and re-exports, because that is what attribute lookup sees.

    The third value says the module's surface cannot be decided statically (it star-imports, or
    defines `__getattr__`). Callers must not flag a miss against such a module.
    """
    defs, bound = set(), set()
    opaque = False
    for n in tree.body:
        if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            bound.add(n.name)
            if not n.name.startswith("_"):
                defs.add(n.name)
            if n.name == "__getattr__":
                opaque = True
        elif isinstance(n, ast.Assign):
            for tgt in n.targets:
                for sub in ast.walk(tgt):
                    if isinstance(sub, ast.Name):
                        bound.add(sub.id)
        elif isinstance(n, ast.AnnAssign):
            if isinstance(n.target, ast.Name):
                bound.add(n.target.id)
        elif isinstance(n, ast.ImportFrom):
            for alias in n.names:
                if alias.name == "*":
                    opaque = True
                else:
                    bound.add(alias.asname or alias.name)
        elif isinstance(n, ast.Import):
            for alias in n.names:
                bound.add(alias.asname or alias.name.split(".")[0])
        elif isinstance(n, (ast.If, ast.Try)):
            # Conditional re-export blocks are common in package __init__ files; treat every name
            # bound in either branch as available rather than guessing which branch is live.
            for sub in ast.walk(n):
                if isinstance(sub, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                    bound.add(sub.name)
                elif isinstance(sub, ast.alias):
                    if sub.name == "*":
                        opaque = True
                    else:
                        bound.add(sub.asname or sub.name.split(".")[0])
    return defs, bound, opaque


def scan_tree(root):
    """{module: (defs, bound, opaque)} for the flydsl package under `root`."""
    pkg_root = os.path.join(root, "flydsl")
    out = {}
    for dirpath, dirnames, filenames in os.walk(pkg_root):
        dirnames[:] = [d for d in dirnames if d != "__pycache__"]
        for fn in sorted(filenames):
            if not fn.endswith(".py"):
                continue
            path = os.path.join(dirpath, fn)
            mod = module_name(pkg_root, path)
            try:
                tree = ast.parse(read_text(path))
            except SyntaxError as e:
                print(f"  ! skipped {path}: {e}", file=sys.stderr)
                continue
            out[mod] = _module_facts(tree)
    return out


# --------------------------------------------------------------------------------------------------
# Diffing
# --------------------------------------------------------------------------------------------------

def version_key(v):
    return tuple(int(p) if p.isdigit() else 0 for p in re.split(r"[.\-+]", v)[:3])


def symbol_index(trees):
    """{symbol: {version: sorted[modules that define it]}} across every scanned version."""
    idx = {}
    for ver, mods in trees.items():
        for mod, (defs, _bound, _opaque) in mods.items():
            for name in defs:
                idx.setdefault(name, {}).setdefault(ver, []).append(mod)
    for name, per_ver in idx.items():
        for ver in per_ver:
            per_ver[ver] = sorted(per_ver[ver])
    return idx


def classify(idx, old, new):
    """Split the symbol index into removed / added / moved between two versions."""
    removed, added, moved = [], [], []
    for name, per_ver in sorted(idx.items()):
        a, b = per_ver.get(old), per_ver.get(new)
        if a and not b:
            removed.append((name, a))
        elif b and not a:
            added.append((name, b))
        elif a and b and a != b:
            moved.append((name, a, b))
    return removed, added, moved


# --------------------------------------------------------------------------------------------------
# Hand-written replacement notes, re-validated against the scan on every run
# --------------------------------------------------------------------------------------------------

def load_notes(path, idx, versions):
    """(entries, errors). An entry survives only if the scan still agrees with what it claims.

    A note that says "X is gone, use Y" is a porting instruction. If X came back or Y went away the
    instruction is wrong, and a wrong instruction is worse than none -- so a stale entry is dropped
    from the output and reported as an error rather than printed with a caveat.
    """
    if not os.path.isfile(path):
        return [], []
    try:
        import yaml
    except ImportError:
        return [], [f"{path}: PyYAML not available, replacement notes skipped"]
    try:
        doc = yaml.safe_load(read_text(path)) or {}
    except Exception as e:  # noqa: BLE001 - any parse failure must degrade to "no notes", not abort
        return [], [f"{path}: unreadable ({e})"]

    kept, errors = [], []
    for i, e in enumerate(doc.get("replacements") or []):
        where = f"{os.path.basename(path)} entry {i + 1} ({e.get('topic', 'no topic')})"
        old, new = str(e.get("from", "")), str(e.get("to", ""))
        if old not in versions or new not in versions:
            errors.append(f"{where}: {old} -> {new} is not a scanned version pair")
            continue
        bad = []
        for sym in e.get("gone") or []:
            if idx.get(sym, {}).get(new):
                bad.append(f"`{sym}` is listed as gone but {new} defines it")
        for sym in e.get("use") or []:
            if not idx.get(sym, {}).get(new):
                bad.append(f"`{sym}` is listed as the replacement but {new} has no such symbol")
        if bad:
            errors.extend(f"{where}: {b}" for b in bad)
            continue
        kept.append(e)
    return kept, errors


# --------------------------------------------------------------------------------------------------
# Import checking -- find call sites that break on the installed version
# --------------------------------------------------------------------------------------------------

def collect_flydsl_imports(path):
    """[(module, symbol_or_None, lineno)] for every flydsl import in one file."""
    try:
        tree = ast.parse(read_text(path))
    except SyntaxError:
        return []
    hits = []
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            if not node.module or not (node.module == "flydsl" or node.module.startswith("flydsl.")):
                continue
            for alias in node.names:
                hits.append((node.module, alias.name, node.lineno))
        elif isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name == "flydsl" or alias.name.startswith("flydsl."):
                    hits.append((alias.name, None, node.lineno))
    return hits


def check_call_sites(dirs):
    """Imports under `dirs` that fail against the flydsl installed in THIS interpreter.

    Resolved live, not statically. Star-imports, `__getattr__`, conditional re-exports and
    `.so`-backed submodules all make a static answer wrong in both directions -- measured here, a
    static resolver produced 56 false positives (compiled `flydsl._mlir` submodules read as absent)
    and 23 false negatives (a missing name forgiven because its package star-imports). The question
    this section answers is "does my code import on the version on this box", and the box can answer
    it exactly.

    Returns (findings, version) where findings is [(path, lineno, module, symbol, reason)].
    """
    # Probing an import runs module-level code, which can raise anything at all -- a bare
    # ModuleNotFoundError filter would let a broken dependency masquerade as a passing import.
    try:
        import flydsl
    except Exception as e:  # noqa: BLE001 - reporting why is the point; re-raising loses the run
        return [], f"not importable ({type(e).__name__})"
    installed = getattr(flydsl, "__version__", "unknown")

    cache = {}

    def resolve(mod, sym):
        if mod not in cache:
            try:
                cache[mod] = importlib.import_module(mod)
            except Exception as e:  # noqa: BLE001 - see above
                cache[mod] = e
        m = cache[mod]
        if isinstance(m, Exception):
            return f"module does not import ({type(m).__name__})"
        if sym is None or hasattr(m, sym):
            return None
        try:
            importlib.import_module(f"{mod}.{sym}")
            return None
        except Exception:  # noqa: BLE001 - see above
            return "name absent from module"

    findings = []
    for d in dirs:
        for dirpath, dirnames, filenames in os.walk(d):
            dirnames[:] = [x for x in dirnames if x != "__pycache__"]
            for fn in sorted(filenames):
                if not fn.endswith(".py"):
                    continue
                path = os.path.join(dirpath, fn)
                for mod, sym, lineno in collect_flydsl_imports(path):
                    why = resolve(mod, sym)
                    if why:
                        findings.append((path, lineno, mod, sym, why))
    return sorted(set(findings)), installed


# --------------------------------------------------------------------------------------------------
# Rendering
# --------------------------------------------------------------------------------------------------

def fmt_mods(mods):
    return ", ".join(f"`{m}`" for m in mods) if mods else "—"


def render(trees, roots, idx, scanned, scan_dirs, installed, notes):
    vers = list(trees.keys())
    L = []
    A = L.append

    A("# FlyDSL version map — where each symbol lives, per version")
    A("")
    A("> **Generated file.** Rebuild with `python3 languages/flydsl/_gen_version_map.py` (see the")
    A("> script docstring for the `--root` arguments). Do not hand-edit the tables.")
    A(">")
    # Sibling generated artifacts (corpus/gemm_decisions.md, corpus/gemm_source_evidence.md and
    # index/run_recurrence.md) DO have a --check
    # gate in CI, so a reader who has seen those will assume this one is held current too. It cannot
    # be: regenerating needs four FlyDSL versions installed side by side, and three of those trees
    # were unpacked wheels that no longer exist. Better to say so here than to let the reader infer a
    # gate that does not exist.
    A("> **Unlike the other generated files here, no CI gate can hold this one current** — rebuilding")
    A("> needs four FlyDSL versions installed side by side, which no CI image has. Treat the version")
    A("> list below as the scope of what was compared, and re-run locally when a new version lands.")
    A("")
    A("A recipe carries a **logic** half (tiling, what to fuse, what goes to LDS, layout, which MFMA)")
    A("and an **API** half (which module to call). The logic half is bound to the architecture and")
    A("survives a version bump; the API half is not. This file resolves the API half mechanically so")
    A("porting a recipe to another FlyDSL version is a lookup instead of a search.")
    A("")
    A("**A version difference is not a reason to discard a recipe.** Reuse the logic, re-derive the")
    A("call form from the table below, then re-measure — a performance number from another version is")
    A("not evidence for this one.")
    A("")

    A("## Versions scanned")
    A("")
    A("Trees are the paths this table was built from, recorded for provenance. Only the installed")
    A("one is expected to persist -- the others were unpacked wheels, so the tables here are the")
    A("durable artifact, not the trees. Symbol counts include the generated `_mlir/dialects/*_ops_gen`")
    A("bindings, which is why they are large; the per-pair diffs below are the part that matters.")
    A("")
    A("| version | tree | modules | public symbols |")
    A("|---|---|---|---|")
    for v in vers:
        mods = trees[v]
        n_syms = sum(len(defs) for defs, _b, _o in mods.values())
        A(f"| `{v}` | `{roots[v]}` | {len(mods)} | {n_syms} |")
    A("")

    for old, new in zip(vers, vers[1:]):
        removed, added, moved = classify(idx, old, new)
        A(f"## {old} → {new}")
        A("")
        A(f"{len(removed)} symbol(s) gone, {len(moved)} moved, {len(added)} new.")
        A("")

        pair_notes = [n for n in notes if str(n.get("from")) == old and str(n.get("to")) == new]
        if pair_notes:
            A("### Known replacements — what carries the old job")
            A("")
            A("Hand-written and re-validated on every regeneration against the tables below")
            A("(`version_map_notes.yaml`). These say what the mechanical diff cannot: which new")
            A("symbol does the removed one's work, and what changed about the call.")
            A("")
            for n in pair_notes:
                A(f"**{n.get('topic', 'untitled')}**")
                A("")
                A(f"- Gone: {', '.join(f'`{s}`' for s in n.get('gone') or []) or '—'}")
                A(f"- Use: {', '.join(f'`{s}`' for s in n.get('use') or []) or '—'}")
                A("")
                A(" ".join(str(n.get("note", "")).split()))
                A("")
                for ev in n.get("evidence") or []:
                    A(f"  - {ev}")
                A("")

        if removed:
            A(f"### Gone in {new} — no definition anywhere in the package")
            A("")
            A("Code using these needs a different way to express the same operation. There is no")
            A("call-form substitution; the logic has to be re-expressed with what the version offers.")
            A("")
            A(f"| symbol | defined in {old} |")
            A("|---|---|")
            for name, mods in removed:
                A(f"| `{name}` | {fmt_mods(mods)} |")
            A("")

        if moved:
            A(f"### Moved in {new} — same name, different module")
            A("")
            A("A move is mechanical only when the new home is the same *kind* of module. When a helper")
            A("moved into an intrinsic namespace, the wrapper's behaviour moved to the caller: verify")
            A("what the old wrapper did before treating the new location as a drop-in.")
            A("")
            A(f"| symbol | {old} | {new} |")
            A("|---|---|---|")
            for name, a, b in moved:
                A(f"| `{name}` | {fmt_mods(a)} | {fmt_mods(b)} |")
            A("")

        gone_mods = sorted(set(trees[old]) - set(trees[new]))
        new_mods = sorted(set(trees[new]) - set(trees[old]))
        if gone_mods or new_mods:
            A("### Modules")
            A("")
            if gone_mods:
                A(f"Removed in {new}: " + ", ".join(f"`{m}`" for m in gone_mods))
                A("")
            if new_mods:
                A(f"Added in {new}: " + ", ".join(f"`{m}`" for m in new_mods))
                A("")

        if added:
            A(f"<details><summary>{len(added)} symbol(s) new in {new}</summary>")
            A("")
            A(f"| symbol | {new} |")
            A("|---|---|")
            for name, mods in added:
                A(f"| `{name}` | {fmt_mods(mods)} |")
            A("")
            A("</details>")
            A("")

    if scan_dirs:
        A(f"## Call sites that fail on the installed FlyDSL (`{installed}`)")
        A("")
        A("Every `flydsl` import under the scanned trees, resolved **live** against the version")
        A("installed in the interpreter that generated this file. Each row raises at import time.")
        A("")
        A("Scanned: " + ", ".join(f"`{d}`" for d in scan_dirs))
        A("")
        if scanned:
            A("| file | line | import | reason |")
            A("|---|---|---|---|")
            for path, lineno, mod, sym, why in scanned:
                target = f"from {mod} import {sym}" if sym else f"import {mod}"
                A(f"| `{path}` | {lineno} | `{target}` | {why} |")
        else:
            A(f"None — every flydsl import under the scanned trees resolves on `{installed}`.")
        A("")

    A("## Sources")
    A("")
    for v in vers:
        A(f"- FlyDSL `{v}` package tree: `{roots[v]}/flydsl/` (`__version__` read from its `__init__.py`)")
    for d in scan_dirs:
        A(f"- Import call sites resolved live against installed FlyDSL `{installed}`: `{d}`")
    A("- Generated by `languages/flydsl/_gen_version_map.py` from the trees above.")
    return "\n".join(L) + "\n"


# --------------------------------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", action="append", default=[], metavar="DIR",
                    help="directory containing a flydsl/ package; repeatable, >=2 to produce a diff")
    ap.add_argument("--check-imports", action="append", default=[], metavar="DIR",
                    help="tree of .py call sites to resolve live against the INSTALLED flydsl")
    ap.add_argument("--notes", default=NOTES,
                    help="hand-written replacement notes, re-validated against the scan")
    ap.add_argument("--out", default=DEFAULT_OUT, help=f"output path (default {DEFAULT_OUT})")
    a = ap.parse_args()

    if len(a.root) < 2:
        ap.error("need at least two --root trees to diff")

    # A path that does not exist silently yields "no failing imports", which reads exactly like a
    # clean result. Refuse instead: a typo here would publish a false all-clear.
    missing = [d for d in a.check_imports if not os.path.isdir(d)]
    if missing:
        ap.error("--check-imports path(s) do not exist: " + ", ".join(missing))

    found = {}
    roots = {}
    for root in a.root:
        ver = read_version(root)
        if ver is None:
            print(f"! {root}: no flydsl/__init__.py with __version__ — skipped", file=sys.stderr)
            continue
        if ver in found:
            print(f"! {root}: version {ver} already scanned from {roots[ver]} — skipped",
                  file=sys.stderr)
            continue
        print(f"scanning {ver} at {root}")
        found[ver] = scan_tree(root)
        roots[ver] = root

    if len(found) < 2:
        print("! fewer than two usable trees; nothing to diff", file=sys.stderr)
        return 1

    trees = OrderedDict(sorted(found.items(), key=lambda kv: version_key(kv[0])))
    roots = {v: roots[v] for v in trees}
    idx = symbol_index(trees)

    notes, note_errors = load_notes(a.notes, idx, set(trees))
    scanned, installed = check_call_sites(a.check_imports) if a.check_imports else ([], "n/a")

    text = render(trees, roots, idx, scanned, a.check_imports, installed, notes)
    with open(a.out, "w", encoding="utf-8") as f:
        f.write(text)

    print(f"wrote {a.out} ({len(text.splitlines())} lines)")
    for old, new in zip(list(trees), list(trees)[1:]):
        removed, added, moved = classify(idx, old, new)
        print(f"  {old} -> {new}: {len(removed)} gone, {len(moved)} moved, {len(added)} new")
    print(f"  replacement notes: {len(notes)} valid, {len(note_errors)} rejected")
    if a.check_imports:
        print(f"  imports failing on installed {installed}: {len(scanned)}")

    for err in note_errors:
        print(f"! {err}", file=sys.stderr)
    return 1 if note_errors else 0


if __name__ == "__main__":
    sys.exit(main())
