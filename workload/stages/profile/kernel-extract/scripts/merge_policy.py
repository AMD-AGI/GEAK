#!/usr/bin/env python3
"""Decide, from the profile evidence of a multi-config group, which tasks are one UT.

    python3 merge_policy.py --group-dir "$OUT" [--baseline "$RUN_ROOT/baseline/out/baseline.json"] \
        [--source-root /sgl-workspace/sglang/python=/path/to/the/same/tree] --out "$OUT/merge_policy.json"

The rule (stages/profile/README.md "Merging across concurrency"):

1. SAME LAUNCH GROUP (same EP/TP/backend...), SAME ENTRY: one UT by default. Every config's
   cases go into it; an M-dependent path INSIDE or BELOW the entry is reached by the cases
   themselves and checked by the `launches` gate. Nothing to decide -- this script only
   reports the configs each task pools.
2. Exception A -- a configuration difference the task's code reads. `baseline.json`
   `launch_diff` lists what differs between the group's configs. A difference splits a task
   only if the task's own source reads it (an `env:` variable named in the task's
   kernel_src or in the entry module). Server flags (`--chunked-prefill-size`, graph bucket
   limits, cache sizes, ...) change which M is served, which the cases already cover.
3. Exception B -- the path is chosen ABOVE the entry, so different M reach different entries
   (different tasks). Such tasks form a BRANCH GROUP when all of these hold:
     - the nearest common caller of their entries is within --max-depth frames of each entry,
       and they call it from different source lines;
     - no sampled point launches two of them (mutually exclusive in M -- co-occurring tasks
       are sequential operators or different layer types, never branches);
     - each is observed at some point.
   A branch group is, by default, a GROUP GATE: the members stay separate optimisation tasks,
   and every member's kept result must not regress any member's cases. A false group costs
   one redundant check and changes no result. It becomes ONE TASK (`merge_task`) only if the
   common caller is itself a replayable entry -- every recorded argument a tensor, scalar,
   sequence or literal (no object, no `self`) -- AND its source has one if/elif chain whose
   arms hold the members' call lines and whose test reads a size. The merged task's seam is
   then a function the deployment already has; nothing is invented.
4. Different launch groups (EP...): same entry and same env -> one UT, cases tagged by group;
   otherwise separate. Reported in `cross_group`.

Stdlib only. Reads: task_plan.json, configs/*/evidence/{source_map,compile_variants}.json,
configs/*/evidence/census/*.json, tasks/*/ (kernel_src, meta.json).
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


def replayable(args):
    """(ok, why) for a recorded frame's arguments."""
    def ok(name, d):
        if isinstance(d, dict) and d.get("__tensor__"):
            return None
        if isinstance(d, dict) and "__seq__" in d:
            for i, v in enumerate(d.get("items") or []):
                w = ok("%s[%d]" % (name, i), v)
                if w:
                    return w
            return None
        if isinstance(d, dict) and "__object__" in d:
            return None if literal(d)[0] else "%s is a %s" % (name, d["__object__"])
        if isinstance(d, dict) and "__missing__" in d:
            return "%s was not bound" % name
        return None
    if not args:
        return False, "no recorded call of this frame"
    if "self" in args:
        return False, "bound method (self is %s)" % (args["self"] or {}).get("__object__", "?")
    for n, d in args.items():
        w = ok(n, d)
        if w:
            return False, w
    return True, "every argument is a tensor, scalar, sequence or literal"


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


def guard_proof(path, lines):
    """(ok, guards): every member's call lines sit under its own `if` whose test reads a size,
    and no member's call is under another member's guard.

    Both shapes a dispatcher takes are accepted: one if/elif chain, and a run of sibling `if
    ...: return` blocks. Mutual exclusivity itself is the evidence's (no sampled point launches
    two members); the source only has to show the choice is made here, on a size."""
    try:
        src = open(path).read()
        tree = ast.parse(src)
    except (OSError, SyntaxError):
        return False, ["source not readable: %s" % path]
    ifs = [n for n in ast.walk(tree) if isinstance(n, ast.If)]
    # Names bound from a size anywhere in the file (`n = x.shape[0]`, `m, k = a.shape`,
    # `num_seqs = len(seqused_k)`): a guard on them reads a size too.
    sized = set()
    for n in ast.walk(tree):
        if isinstance(n, (ast.Assign, ast.AnnAssign)) and n.value is not None \
                and SIZE_RE.search(ast.get_source_segment(src, n.value) or ""):
            for t in (n.targets if isinstance(n, ast.Assign) else [n.target]):
                sized |= {x.id for x in ast.walk(t) if isinstance(x, ast.Name)}

    def reads_size(test):
        text = ast.get_source_segment(src, test) or ""
        return bool(SIZE_RE.search(text)) or any(
            isinstance(x, ast.Name) and x.id in sized for x in ast.walk(test))

    def body_span(stmts):
        return min(s.lineno for s in stmts), max(getattr(s, "end_lineno", s.lineno) for s in stmts)

    def guard(line):
        """The innermost `if` whose taken body (not its else) holds the line."""
        best = None
        for n in ifs:
            lo, hi = body_span(n.body)
            if lo <= line <= hi and (best is None or n.lineno > best.lineno):
                best = n
        return best

    guards, texts = [], []
    for member_lines in lines:
        gs = [guard(ln) for ln in sorted(member_lines)]
        if any(g is None for g in gs):
            return False, ["a call line of a member is not under any if"]
        tests = [ast.get_source_segment(src, g.test) or "" for g in gs]
        if not all(reads_size(g.test) for g in gs):
            return False, ["a member's guard does not read a size: %s" % " | ".join(tests)]
        guards.append(gs)
        texts += tests
    for i, gs in enumerate(guards):
        for j, other in enumerate(lines):
            if i == j:
                continue
            for g in gs:
                lo, hi = body_span(g.body)
                if any(lo <= ln <= hi for ln in other):
                    return False, ["a member's call is inside another member's guard (line %d)" % g.lineno]
    return True, texts


# ---------------------------------------------------------------- the policy
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


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--group-dir", required=True, help="the profile stage's $OUT")
    ap.add_argument("--baseline", help="baseline.json (its launch_diff)")
    ap.add_argument("--source-root", action="append", default=[],
                    help="OLD=NEW: read the deployment's source at NEW where evidence says OLD")
    ap.add_argument("--max-depth", type=int, default=2,
                    help="frames above an entry to search for the common caller")
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

    # callers above each task's entry
    callers, entry_path = {}, {}
    for n, t in tasks.items():
        for k in kernels_of[n]:
            fr = stk.get(k)
            if not fr:
                continue
            idx = next((i for i, f in enumerate(fr) if f[0] == t["target_callable"]), None)
            if idx is None:
                continue
            # (callable, path, def line, callee name called from it)
            chain = fr[idx: idx + 1 + a.max_depth]
            callers[n] = [(c[0], c[1], c[2], chain[j][0].split(":")[-1].split(".")[-1])
                          for j, c in enumerate(chain[1:])]
            entry_path[n] = remap(fr[idx][1], roots)
            break

    # ---- rule 1 + exception A
    diff = (load(a.baseline, {}) or {}).get("launch_diff") or {} if a.baseline else {}
    per_task = {}
    for n, t in tasks.items():
        cfg_set = sorted({p.split("__", 1)[0] for p in t.get("owned_point_ids") or []})
        files = task_sources(t.get("task_dir") or os.path.join(g, "tasks", n), entry_path.get(n))
        reads = {}
        for grp, keys in diff.items():
            for key in keys:
                if key.startswith("env:"):
                    hits = env_reads(files, key[4:])
                    if hits:
                        reads[key] = {"group": grp, "values": keys[key],
                                      "read_by": [os.path.relpath(h, g) if h.startswith(g) else h
                                                  for h in hits]}
        per_task[n] = {"configs": cfg_set, "decision": "split_by_config_value" if reads else "merge",
                       "reads_config_difference": reads,
                       "why": ("its source reads %s, which differs between configs: one UT per value"
                               % ", ".join(sorted(reads)) if reads else
                               "same entry in every config; no differing setting is read by its source")}

    # ---- exception B: branch groups
    names = sorted(callers)
    edges, pair_info = defaultdict(set), {}
    for i, x in enumerate(names):
        for y in names[i + 1:]:
            fx, fy = callers[x], callers[y]
            cy = [f[0] for f in fy]
            ix = next((j for j, f in enumerate(fx) if f[0] in cy), None)
            if ix is None:
                continue
            iy = cy.index(fx[ix][0])
            phases = set(tasks[x].get("phases") or []) & set(tasks[y].get("phases") or [])
            # exclusivity is only meaningful inside one phase
            px = {(c, p) for c, p, ph, _s in pres.get(x, ()) if ph in phases}
            py = {(c, p) for c, p, ph, _s in pres.get(y, ()) if ph in phases}
            if not phases or not px or not py or px & py or fx[ix][3] == fy[iy][3]:
                continue
            path = remap(fx[ix][1], roots)
            lx = call_lines(path, fx[ix][2], fx[ix][3])
            ly = call_lines(path, fy[iy][2], fy[iy][3])
            if not lx or not ly or set(lx) & set(ly):
                continue
            edges[x].add(y)
            edges[y].add(x)
            pair_info[(x, y)] = (fx[ix][0], path, lx, ly)
    # Groups are formed per common caller, so a caller high up (a layer's forward) cannot chain
    # unrelated operators together through a shared member.
    by_caller = defaultdict(dict)
    for (x, y), info in pair_info.items():
        by_caller[info[0]][(x, y)] = info
    groups = []
    for caller, pairs in sorted(by_caller.items()):
        adj = defaultdict(set)
        for x, y in pairs:
            adj[x].add(y)
            adj[y].add(x)
        seen = set()
        for n in sorted(adj):
            if n in seen:
                continue
            comp, todo = set(), [n]
            while todo:
                m = todo.pop()
                if m not in comp:
                    comp.add(m)
                    todo += list(adj[m])
            seen |= comp
            members = sorted(comp)
            cpath = next(iter(pairs.values()))[1]
            lines = {}
            for (x, y), (_c, _p, lx, ly) in pairs.items():
                if x in comp:
                    lines[x], lines[y] = lx, ly
            shapes = {m: sorted({s for _c, _p, _ph, s in pres[m] if s is not None}) for m in members}
            complete = all((p, q) in pairs or (q, p) in pairs
                           for k, p in enumerate(members) for q in members[k + 1:])
            rep_ok, rep_why = replayable(rec.get(caller))
            prf_ok, tests = guard_proof(cpath, [set(lines[m]) for m in members])
            tier = "merge_task" if complete and rep_ok and prf_ok else "group_gate"
            reasons = [w for w, o in ((
                "not every pair is mutually exclusive", complete), (rep_why, rep_ok), (
                "; ".join(tests), prf_ok)) if not o]
            groups.append({
                "members": members, "common_caller": caller, "caller_file": cpath,
                "call_lines": lines, "observed_shapes": shapes, "tier": tier,
                "pairwise_exclusive": complete, "replayable": rep_ok, "replayable_why": rep_why,
                "guards_read_size": prf_ok, "guards": tests,
                "why": ("the common caller is a replayable entry that picks the member by a size "
                        "test: one task whose seam is %s" % caller if tier == "merge_task" else
                        "members stay separate tasks; each kept result must pass every member's "
                        "cases (%s)" % "; ".join(reasons))})

    # ---- different launch groups
    groups_of_cfg = defaultdict(set)
    for grp, keys in diff.items():
        for vals in keys.values():
            for c in vals:
                groups_of_cfg[grp].add(c)
    cross = {"launch_groups": sorted(groups_of_cfg) or ["(one)"],
             "note": ("one launch group: nothing to merge across groups"
                      if len(groups_of_cfg) <= 1 else
                      "same entry and same env across groups -> one UT with cases tagged by group; "
                      "otherwise separate")}

    doc = {"contract": "kernel_extract/merge_policy/v1", "group_dir": os.path.abspath(g),
           "configs": cfgs, "per_task": per_task, "branch_groups": groups,
           "cross_group": cross,
           "unplaced_tasks": sorted(set(tasks) - set(callers))}
    with open(a.out, "w") as fh:
        json.dump(doc, fh, indent=1)
    for gr in groups:
        print("%-11s %s%s" % (gr["tier"], " + ".join(gr["members"]),
                              "  @ %s" % gr.get("common_caller") if gr.get("common_caller") else ""))
    split = [n for n, v in per_task.items() if v["decision"] != "merge"]
    print("per-task: %d merge across configs, %d split by a setting they read%s"
          % (len(per_task) - len(split), len(split), (" (%s)" % ", ".join(split)) if split else ""))
    print("-> %s" % a.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
