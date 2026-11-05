#!/usr/bin/env python3
"""Decide, from the profile evidence of a multi-config group, which tasks are one UT.

    python3 merge_policy.py --group-dir "$OUT" [--baseline "$RUN_ROOT/baseline/out/baseline.json"] \
        [--source-root /sgl-workspace/sglang/python=/path/to/the/same/tree] --out "$OUT/merge_policy.json"

The rule (stages/profile/README.md "Merging across concurrency"): an operator dispatched by
batch size is ONE UT. Whether it is one is decided strictly; once it is, it is merged --
how hard the harness is to write never keeps it apart.

1. Same launch group, same entry: one UT whose cases pool every config (M-dependent paths
   inside or below the entry are reached by the cases). Reported in `per_task`.
2. A `launch_diff` setting splits a task only if the task's own source reads it.
3. Different entries chosen ABOVE them by batch size -- a BRANCH GROUP -- when, for a pair:
     - their nearest common caller (<= --max-depth frames up) calls them from different lines;
     - each call sits under a conditional (if / elif / conditional expression) in that caller;
     - no sampled point of a shared phase launches both, and their batch sizes do not
       interleave (all of one below all of the other).
   Pairs link transitively; the group's SEAM is the outermost caller on every member's stack.
   HOW it is merged follows from the seam's recorded arguments, never whether:
     dispatcher        every argument replayable (tensor, scalar, sequence, literal):
                       the seam is the UT entry; its threshold is part of the task
     rebuilt_instance  only `self` is not: the UT entry is the seam on an instance rebuilt
                       from `attribute_manifest` (every self.<attr> the seam and the helpers
                       it hands `self` to read)
     scope_equalized   something else is not (a ForwardBatch ...): one task with one entry per
                       member; each case calls its member's entry; `scope_region` lists the
                       calls between the members' calls and where their paths rejoin, which the
                       narrower member's case must also run so both produce the same output
4. Different launch groups (EP, ...): reported in `cross_group`.

Every merged group carries what its UT needs to be checked branch by branch: members and
their expected kernels, the condition each sits under, the batch sizes observed per member,
the boundary between them, which cases come from which member, and whether the threshold
may move.

Stdlib only. Reads: task_plan.json, configs/*/evidence/{source_map,compile_variants}.json,
configs/*/evidence/census/*.json, tasks/*/kernel_src, and the deployment's source.
"""
import argparse
import ast
import glob
import json
import os
import re
import sys
from collections import defaultdict

SIZE_RE = re.compile(r"shape|size|numel|len\(|num_|rows|batch|\bbs\b|\bm\b|seq|tokens", re.I)


def load(path, default=None):
    try:
        with open(path) as fh:
            return json.load(fh)
    except (OSError, ValueError):
        return default


# ---------------------------------------------------------------- evidence
def configs(group):
    return sorted(d for d in os.listdir(os.path.join(group, "configs"))
                  if os.path.isdir(os.path.join(group, "configs", d, "evidence")))


def stacks(group, cfgs):
    """kernel -> [(callable, path, line), ...] innermost first, from the first config that has it."""
    out = {}
    for c in cfgs:
        sm = load(os.path.join(group, "configs", c, "evidence", "source_map.json"), {})
        for k, v in (sm.get("kernels") or {}).items():
            fr = [(f.get("target_callable"), f.get("path"), f.get("line"))
                  for f in v.get("frames") or [] if f.get("target_callable")]
            if fr and k not in out:
                out[k] = fr
    return out


def recorded_args(group, cfgs):
    """callable -> recorded argument descriptors of one frame (compile_watch evidence)."""
    out = {}
    for c in cfgs:
        cv = load(os.path.join(group, "configs", c, "evidence", "compile_variants.json"), {})
        for v in cv.get("variants") or []:
            for f in v.get("frames") or []:
                out.setdefault(f.get("callable"), f.get("args") or {})
    return out


def presence(group, cfgs, kernels_of):
    """task -> {(config, point, phase, shape)} where any of its kernels ran (rank 0)."""
    out = defaultdict(set)
    for c in cfgs:
        for f in glob.glob(os.path.join(group, "configs", c, "evidence", "census", "*.json")):
            d = load(f, {})
            row = (d.get("per_rank") or {}).get("0") or {}
            if not row:
                continue
            key = (c, d.get("point_id"), d.get("phase"), d.get("achieved_shape"))
            for t, ks in kernels_of.items():
                if any(row.get(k) for k in ks):
                    out[t].add(key)
    return out


# ---------------------------------------------------------------- replayability / AST
def literal(d):
    """A recorded builtin container (frozenset/set/tuple/dict of literals) rebuilt from its repr."""
    kind = d.get("__object__") if isinstance(d, dict) else None
    if kind not in ("builtins.frozenset", "builtins.set", "builtins.dict", "builtins.tuple"):
        return False, None
    text = (d.get("repr") or "").strip()
    if len(text) >= 120:                        # compile_watch truncates reprs at 120 characters
        return False, None
    m = re.fullmatch(r"(frozenset|set)\((.*)\)", text, re.S)
    try:
        if m:
            inner = m.group(2).strip()
            items = ast.literal_eval(inner) if inner else ()
            return True, (frozenset if m.group(1) == "frozenset" else set)(items)
        return True, ast.literal_eval(text)
    except (ValueError, SyntaxError):
        return False, None


def remap(path, roots):
    for old, new in roots:
        if path and path.startswith(old):
            return new + path[len(old):]
    return path


def call_lines(path, func_line, callee):
    """Lines in the function defined at `func_line` of `path` that call a name `callee`.

    The profiler's stack frames carry the line a function is DEFINED at, not the line it
    was called from, so the call sites are read from the caller's source."""
    try:
        tree = ast.parse(open(path).read())
    except (OSError, SyntaxError):
        return []
    fn = next((n for n in ast.walk(tree)
               if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)) and n.lineno == func_line), None)
    if fn is None:
        return []
    out = set()
    for n in ast.walk(fn):
        if isinstance(n, ast.Call):
            f = n.func
            name = f.id if isinstance(f, ast.Name) else f.attr if isinstance(f, ast.Attribute) else None
            if name == callee:
                out.add(n.lineno)
    return sorted(out)


def task_sources(task_dir, entry_path):
    files = glob.glob(os.path.join(task_dir, "kernel_src", "**", "*.py"), recursive=True)
    if entry_path and os.path.exists(entry_path):
        files.append(entry_path)
    return files


def env_reads(files, var):
    pat = re.compile(r"\b%s\b" % re.escape(var))
    hits = []
    for f in files:
        try:
            if pat.search(open(f, errors="replace").read()):
                hits.append(f)
        except OSError:
            pass
    return hits




def replayable(args):
    """(kind, why): 'all' | 'self_only' | 'no' for a recorded frame's arguments."""
    def bad(name, d):
        if isinstance(d, dict) and d.get("__tensor__"):
            return None
        if isinstance(d, dict) and "__seq__" in d:
            for i, v in enumerate(d.get("items") or []):
                w = bad("%s[%d]" % (name, i), v)
                if w:
                    return w
            return None
        if isinstance(d, dict) and "__object__" in d:
            return None if literal(d)[0] else "%s is a %s" % (name, d["__object__"])
        if isinstance(d, dict) and "__missing__" in d:
            return "%s was not bound" % name
        return None
    if not args:
        return "no", "no recorded call of the seam"
    others = [w for w in (bad(n, d) for n, d in args.items() if n != "self") if w]
    if others:
        return "no", "; ".join(others)
    if "self" in args:
        return "self_only", "bound method; every other argument is replayable"
    return "all", "every argument is a tensor, scalar, sequence or literal"


# ---------------------------------------------------------------- source analysis
class Source:
    """One file of the deployment's source, with the conditionals around a line."""

    def __init__(self, path):
        self.path = path
        try:
            self.src = open(path).read()
            self.tree = ast.parse(self.src)
        except (OSError, SyntaxError):
            self.src, self.tree = "", None
        self.conds = []                 # (node, arm, lo, hi): arm in body/orelse
        for n in ast.walk(self.tree) if self.tree else ():
            if isinstance(n, ast.If):
                for arm, stmts in (("body", n.body), ("orelse", n.orelse)):
                    if stmts:
                        self.conds.append((n, arm, min(s.lineno for s in stmts),
                                           max(getattr(s, "end_lineno", s.lineno) for s in stmts)))
            elif isinstance(n, ast.IfExp):
                for arm, e in (("body", n.body), ("orelse", n.orelse)):
                    self.conds.append((n, arm, e.lineno, getattr(e, "end_lineno", e.lineno)))
        sized = set()
        for n in ast.walk(self.tree) if self.tree else ():
            if isinstance(n, (ast.Assign, ast.AnnAssign)) and n.value is not None \
                    and SIZE_RE.search(self.seg(n.value)):
                for t in (n.targets if isinstance(n, ast.Assign) else [n.target]):
                    sized |= {x.id for x in ast.walk(t) if isinstance(x, ast.Name)}
        self.sized = sized

    def seg(self, node):
        return ast.get_source_segment(self.src, node) or ""

    def function(self, def_line):
        return next((n for n in ast.walk(self.tree) if self.tree
                     and isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))
                     and n.lineno == def_line), None)

    def enclosing(self, line, within=None):
        """Conditional arms holding `line`, innermost last (optionally inside a function)."""
        lo_f, hi_f = (within.lineno, within.end_lineno) if within else (0, 1 << 30)
        out = [(n, arm, lo, hi) for n, arm, lo, hi in self.conds
               if lo <= line <= hi and lo_f <= n.lineno <= hi_f]
        return sorted(out, key=lambda c: (c[2], -c[3]))

    def guard(self, line, fn):
        """(test text, reads a size) of the innermost conditional holding the line, or None."""
        enc = self.enclosing(line, fn)
        if not enc:
            return None
        n, arm = enc[-1][0], enc[-1][1]
        test = n.test
        text = self.seg(test)
        size = bool(SIZE_RE.search(text)) or any(isinstance(x, ast.Name) and x.id in self.sized
                                                for x in ast.walk(test))
        return ("not (%s)" % text if arm == "orelse" else text), size

    def self_reads(self, fn, param="self", depth=1, roots=()):
        """Every `<param>.<attr>[.<attr>...]` the function reads, plus the same for helpers it
        passes `param` to as the first positional argument (one level, resolved by name in
        this file)."""
        reads = set()
        for n in ast.walk(fn):
            if isinstance(n, ast.Attribute):
                chain, cur = [], n
                while isinstance(cur, ast.Attribute):
                    chain.append(cur.attr)
                    cur = cur.value
                if isinstance(cur, ast.Name) and cur.id == param:
                    reads.add(".".join(reversed(chain)))
        # keep the longest chains only (self.router.weight, not also self.router)
        reads = {r for r in reads if not any(o != r and o.startswith(r + ".") for o in reads)}
        if depth:
            for n in ast.walk(fn):
                if isinstance(n, ast.Call) and n.args and isinstance(n.args[0], ast.Name) \
                        and n.args[0].id == param:
                    name = n.func.attr if isinstance(n.func, ast.Attribute) else getattr(n.func, "id", None)
                    helper = next((h for h in ast.walk(self.tree)
                                   if isinstance(h, ast.FunctionDef) and h.name == name), None)
                    if helper is None:
                        helper = find_def(name, roots)
                    if helper is not None and helper.args.args:
                        p0 = helper.args.args[0].arg
                        reads |= {"%s" % r for r in Source.__dict__["self_reads"](
                            _Holder(helper), helper, p0, 0)}
        return sorted(reads)


class _Holder:
    """Lets self_reads run on a helper parsed from another file."""
    def __init__(self, fn):
        self.tree = fn


_DEF_CACHE = {}


def find_def(name, roots):
    """A top-level function `name` anywhere under the source roots (first match)."""
    if not name or not roots:
        return None
    if name in _DEF_CACHE:
        return _DEF_CACHE[name]
    pat = re.compile(r"^def %s\(" % re.escape(name), re.M)
    for _old, new in roots:
        for f in glob.glob(os.path.join(new, "**", "*.py"), recursive=True):
            try:
                text = open(f, errors="replace").read()
            except OSError:
                continue
            if pat.search(text):
                fn = next((n for n in ast.parse(text).body
                           if isinstance(n, ast.FunctionDef) and n.name == name), None)
                _DEF_CACHE[name] = fn
                return fn
    _DEF_CACHE[name] = None
    return None


def scope_region(source, fn, member_lines):
    """Calls between the first member call and the end of the statement holding the last one,
    except the members' own calls, each with the conditions it runs under."""
    allm = sorted(l for ls in member_lines.values() for l in ls)
    stmts = [s for s in ast.walk(fn) if isinstance(s, ast.stmt) and s is not fn]
    top = [s for s in fn.body]
    first = next(s for s in top if s.lineno <= allm[0] <= s.end_lineno)
    last = next(s for s in top if s.lineno <= allm[-1] <= s.end_lineno)
    lo, hi = first.lineno, last.end_lineno
    out = []
    for n in ast.walk(fn):
        if isinstance(n, ast.Call) and lo <= n.lineno <= hi and n.lineno not in allm:
            name = n.func.attr if isinstance(n.func, ast.Attribute) else getattr(n.func, "id", "?")
            conds = [("not (%s)" % source.seg(c[0].test) if c[1] == "orelse" else source.seg(c[0].test))
                     for c in source.enclosing(n.lineno, fn)]
            out.append({"line": n.lineno, "call": name, "under": conds})
    del stmts
    return {"lines": [lo, hi], "calls": sorted(out, key=lambda r: r["line"])}


def callee_name(inner_callable, caller_path):
    """The name the caller uses to call `inner_callable`. A plain function is called by its
    name; an nn.Module's forward is reached through __call__ on the attribute holding the
    module (`self.router(x)` for `self.router = MuseSpark1xRouter(...)`), found in the
    caller's file. None when that attribute cannot be found."""
    qual = inner_callable.split(":")[-1]
    parts = qual.split(".")
    if parts[-1] not in ("forward", "__call__") or len(parts) < 2:
        return parts[-1]
    cls = parts[-2]
    try:
        text = open(caller_path).read()
    except (OSError, TypeError):
        return None
    m = re.search(r"self\.(\w+)\s*(?::[^=]+)?=\s*%s\(" % re.escape(cls), text)
    return m.group(1) if m else None


# ---------------------------------------------------------------- the policy
def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--group-dir", required=True, help="the profile stage's $OUT")
    ap.add_argument("--baseline", help="baseline.json (its launch_diff)")
    ap.add_argument("--source-root", action="append", default=[],
                    help="OLD=NEW: read the deployment's source at NEW where evidence says OLD")
    ap.add_argument("--max-depth", type=int, default=2,
                    help="frames above an entry to search for the nearest common caller")
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    roots = [tuple(r.split("=", 1)) for r in a.source_root]
    g = a.group_dir
    plan = load(os.path.join(g, "task_plan.json"), {})
    tasks = {t["label"]: t for t in plan.get("tasks") or []}
    cfgs = configs(g)
    stk = stacks(g, cfgs)
    rec = recorded_args(g, cfgs)
    kernels_of = {n: t.get("selected_identities") or [] for n, t in tasks.items()}
    pres = presence(g, cfgs, kernels_of)
    sources = {}

    def source(path):
        if path not in sources:
            sources[path] = Source(path)
        return sources[path]

    # Each task's stack above its entry: [(callable, path, def line, callee name), ...]
    chain, entry_path = {}, {}
    for n, t in tasks.items():
        for k in kernels_of[n]:
            fr = stk.get(k)
            if not fr:
                continue
            idx = next((i for i, f in enumerate(fr) if f[0] == t["target_callable"]), None)
            if idx is None:
                continue
            up = fr[idx:]
            chain[n] = [(c[0], remap(c[1], roots), c[2], callee_name(up[j][0], remap(c[1], roots)))
                        for j, c in enumerate(up[1:])]
            entry_path[n] = remap(fr[idx][1], roots)
            break

    # ---- rule 1 + 2: per task
    diff = ((load(a.baseline, {}) or {}).get("launch_diff") or {}) if a.baseline else {}
    per_task = {}
    for n, t in tasks.items():
        cfg_set = sorted({p.split("__", 1)[0] for p in t.get("owned_point_ids") or []})
        files = task_sources(t.get("task_dir") or os.path.join(g, "tasks", n), entry_path.get(n))
        reads = {}
        for grp, keys in diff.items():
            for key, vals in keys.items():
                if key.startswith("env:"):
                    hits = env_reads(files, key[4:])
                    if hits:
                        reads[key] = {"group": grp, "values": vals, "read_by": hits}
        per_task[n] = {"configs": cfg_set,
                       "decision": "split_by_config_value" if reads else "merge_across_configs",
                       "reads_config_difference": reads}

    # ---- rule 3: link pairs
    def shapes_by_phase(n, phases):
        out = defaultdict(set)
        for _c, _p, ph, s in pres.get(n, ()):
            if ph in phases and s is not None:
                out[ph].add(s)
        return out

    links, refused = {}, []
    names = sorted(chain)
    for i, x in enumerate(names):
        for y in names[i + 1:]:
            fx, fy = chain[x][:a.max_depth], chain[y][:a.max_depth]
            cy = [f[0] for f in fy]
            ix = next((j for j, f in enumerate(fx) if f[0] in cy), None)
            if ix is None:
                continue
            iy = cy.index(fx[ix][0])
            caller, path, dline = fx[ix][0], fx[ix][1], fx[ix][2]
            if fx[ix][3] == fy[iy][3]:
                continue                               # same callee: not two branches here
            if None in (fx[ix][3], fy[iy][3]):
                refused.append({"pair": [x, y], "caller": fx[ix][0],
                                "why": "an entry is a Module.forward reached via __call__ and no "
                                       "`self.<attr> = <Class>(` names its call site"})
                continue
            phases = set(tasks[x].get("phases") or []) & set(tasks[y].get("phases") or [])
            if not phases:
                continue
            src = source(path)
            fn = src.function(dline) if src.tree else None
            if fn is None:
                continue
            lx, ly = call_lines(path, dline, fx[ix][3]), call_lines(path, dline, fy[iy][3])
            if not lx or not ly or set(lx) & set(ly):
                continue
            px = {(c, p) for c, p, ph, _s in pres.get(x, ()) if ph in phases}
            py = {(c, p) for c, p, ph, _s in pres.get(y, ()) if ph in phases}
            sx, sy = shapes_by_phase(x, phases), shapes_by_phase(y, phases)
            why = None
            if not px or not py:
                why = "not both observed in a shared phase"
            elif px & py:
                why = "launched together at %d sampled point(s): not a batch-size dispatch" % len(px & py)
            elif any(sx[ph] and sy[ph] and not (max(sx[ph]) < min(sy[ph]) or max(sy[ph]) < min(sx[ph]))
                     for ph in phases):
                why = "their batch sizes interleave"
            gx = [src.guard(l, fn) for l in lx]
            gy = [src.guard(l, fn) for l in ly]
            if why is None and (None in gx or None in gy):
                why = "a call is not under any conditional in %s" % caller
            if why:
                refused.append({"pair": [x, y], "caller": caller, "why": why})
                continue
            links[(x, y)] = {"caller": caller, "path": path, "def_line": dline,
                             "lines": {x: lx, y: ly}, "guards": {x: gx, y: gy}}

    # ---- groups: transitive closure; seam = outermost caller on every member's stack
    parent = {}

    def find(u):
        parent.setdefault(u, u)
        while parent[u] != u:
            parent[u] = parent[parent[u]]
            u = parent[u]
        return u
    for x, y in links:
        parent[find(x)] = find(y)
    comps = defaultdict(list)
    for u in list(parent):
        comps[find(u)].append(u)

    groups = []
    for members in sorted(sorted(c) for c in comps.values()):
        # the seam is the innermost caller on every member's stack (their nearest common caller)
        seam = next((c for c in chain[members[0]]
                     if all(any(f[0] == c[0] for f in chain[m]) for m in members)), None)
        if seam is None:
            groups.append({"members": members, "merged": False,
                           "why": "linked pairwise but no caller is on every member's stack"})
            continue
        seam_callable, seam_path, seam_line = seam[0], seam[1], seam[2]
        src = source(seam_path)
        fn = src.function(seam_line)
        # each member's call line(s) in the seam, by the callee its stack names there
        lines, guards = {}, {}
        for m in members:
            callee = next(f[3] for f in chain[m] if f[0] == seam_callable)
            if callee is None:
                callee = "?"
            lines[m] = call_lines(seam_path, seam_line, callee)
            guards[m] = [src.guard(l, fn) for l in lines[m]] if fn else []
        kind, kind_why = replayable(rec.get(seam_callable))
        how = {"all": "dispatcher", "self_only": "rebuilt_instance"}.get(kind, "scope_equalized")
        phases = sorted(set.union(*(set(tasks[m].get("phases") or []) for m in members)))
        observed = {m: {ph: sorted(v) for ph, v in shapes_by_phase(m, set(phases)).items()}
                    for m in members}
        # boundaries only between linked (mutually exclusive) members
        boundaries = defaultdict(list)
        for (x, y) in links:
            if x not in members:
                continue
            for ph in phases:
                vx, vy = observed[x].get(ph), observed[y].get(ph)
                if vx and vy:
                    lo, hi = (x, y) if max(vx) < min(vy) else (y, x)
                    boundaries[ph].append({"below": lo, "max_m": max(observed[lo][ph]),
                                           "above": hi, "min_m": min(observed[hi][ph])})
        co = sorted([x, y] for k, x in enumerate(members) for y in members[k + 1:]
                    if {(c, p) for c, p, _ph, _s in pres.get(x, ())}
                    & {(c, p) for c, p, _ph, _s in pres.get(y, ())})
        cases = {m: sorted([{"config": c, "point": p, "phase": ph, "m": s}
                            for c, p, ph, s in pres.get(m, ())],
                           key=lambda r: (r["config"], r["phase"], r["m"] or 0)) for m in members}
        grp = {
            "members": members, "merged": True, "how": how, "how_why": kind_why,
            "seam": seam_callable, "seam_file": seam_path, "seam_def_line": seam_line,
            "threshold_movable": how in ("dispatcher", "rebuilt_instance"),
            "member_entries": {m: tasks[m]["target_callable"] for m in members},
            "expected_kernels": {m: kernels_of[m] for m in members},
            "call_lines": lines,
            "conditions": {m: [x[0] for x in gs if x] for m, gs in guards.items()},
            "conditions_read_size": {m: all(x and x[1] for x in gs) for m, gs in guards.items()},
            "observed_m": observed, "boundaries": dict(boundaries), "case_sources": cases,
            "co_occurring_pairs": co,
            "links": [{"pair": list(k), "caller": v["caller"]} for k, v in links.items()
                      if k[0] in members],
        }
        if how == "rebuilt_instance" and fn is not None:
            grp["attribute_manifest"] = src.self_reads(fn, "self", 1, roots)
        if how == "scope_equalized" and fn is not None:
            grp["scope_region"] = scope_region(src, fn, lines)
        groups.append(grp)

    # ---- rule 4: launch groups
    launch_groups = sorted(diff) or ["(one)"]
    cross = {"launch_groups": launch_groups,
             "note": ("one launch group: nothing to merge across groups" if len(launch_groups) <= 1
                      else "same entry and same env across groups -> one UT with cases tagged by "
                           "group; otherwise separate")}

    merged_into = {m: gi for gi, gr in enumerate(groups) if gr.get("merged") for m in gr["members"]}
    for n, gi in merged_into.items():
        per_task[n]["merged_into"] = "branch_groups[%d] (seam %s)" % (gi, groups[gi]["seam"])
    doc = {"contract": "kernel_extract/merge_policy/v2", "group_dir": os.path.abspath(g),
           "configs": cfgs, "per_task": per_task, "branch_groups": groups,
           "refused_pairs": refused, "cross_group": cross,
           "unplaced_tasks": sorted(set(tasks) - set(chain))}
    with open(a.out, "w") as fh:
        json.dump(doc, fh, indent=1)
    for gr in groups:
        print("%-17s %s  @ %s" % (gr.get("how", "not merged"), " + ".join(gr["members"]),
                                  gr.get("seam", gr.get("why"))))
    split = [n for n, v in per_task.items() if v["decision"] != "merge_across_configs"]
    print("per-task: %d pool all their configs, %d split by a setting they read%s"
          % (len(per_task) - len(split), len(split), (" (%s)" % ", ".join(split)) if split else ""))
    print("-> %s" % a.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
