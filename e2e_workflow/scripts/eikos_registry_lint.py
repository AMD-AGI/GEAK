#!/usr/bin/env python3
"""Lint: every typed field an agent returns in a GEAK workflow is classified in the decision registry.

The registry (eikos_questions/decision_registry.json) is the required declaration for the "consult
Eikos on judgment decisions" rule (design: research/eikos_hard_rule_proposal_20261005.md, rev 2).
This lint does NOT trust the registry to say what exists. It discovers, independently, from the
workflow sources:

  * every agent call site that passes a `schema` (including the `{ schema }` shorthand), with its
    label/phase static prefix and the schema expression it passes;
  * every schema expression's value, by evaluating it in an empty `node` vm context together with
    the `const` definitions it references (obj/arrObj/arrStr fragments, named schemas, inline
    literals, aliases). A reference that cannot be resolved purely -- a runtime value, a function
    parameter, an undefined name -- is an ERROR, never an empty schema;
  * every field path in the resolved schema graph: leaves, array nodes and open-object boundaries
    (`additionalProperties: true`, which every `obj()` schema has).

It then reconciles discovery with the registry, both ways: an unclassified call site or field fails,
and so does a registered one the sources no longer contain. Classification itself is a human decision
recorded in the registry; nothing here infers a class from a JSON type.

What a pass establishes: the registry's inventory matches the sources. It does not establish that
any capture hook or replay receipt exists (the registry says so per decision), nor that judgments made
in free text or inside an agent's own reasoning went through Eikos.

    python3 eikos_registry_lint.py            # lint the three entry files against the registry
    python3 eikos_registry_lint.py --discover # print what discovery found (no registry needed)
"""
import argparse
import json
import os
import re
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
GEAK_ROOT = os.path.dirname(os.path.dirname(HERE))
REGISTRY = os.path.join(HERE, "eikos_questions", "decision_registry.json")
ENTRY_FILES = ("kernel_workflow/kernel_lane.js", "kernel_workflow/kernel_workflow.js",
               "e2e_workflow/e2e_workflow.js")
CLASSES = ("judgment", "mixed", "derived_fact", "action_report", "measurement_report", "identity",
           "artifact_ref", "free_text", "open_object", "unresolved")
NEEDS_DECISION = ("judgment", "mixed", "unresolved")
JS_GLOBALS = {"Object", "JSON", "Array", "String", "Number", "Boolean", "Math", "undefined", "null",
              "true", "false", "Infinity", "NaN"}
REGEX_PREV = set("(,=:[!&|?{};+-*%<>~^") | {"return", "typeof", "instanceof", "in", "of", "new",
                                           "delete", "void", "throw", "case", "do", "else", "=>"}
STATEMENT_START = {"const", "let", "var", "function", "if", "for", "while", "return", "export",
                   "class", "async", "await", "try", "throw", "log", "phase", "do", "switch"}


# --------------------------------------------------------------------------- tokenizer
class Tok:
    __slots__ = ("kind", "text", "start", "end", "line")

    def __init__(self, kind, text, start, end, line):
        self.kind, self.text, self.start, self.end, self.line = kind, text, start, end, line

    def __repr__(self):
        return "%s:%r@%d" % (self.kind, self.text, self.line)


def _skip_string(src, i):
    q = src[i]
    i += 1
    while i < len(src):
        c = src[i]
        if c == "\\":
            i += 2
            continue
        if c == q:
            return i + 1
        if c == "\n" and q != "`":
            raise ValueError("unterminated string")
        i += 1
    raise ValueError("unterminated string")


def _skip_template(src, i):
    """i at the opening backtick; returns the index after the closing one, skipping ${...} bodies."""
    i += 1
    while i < len(src):
        c = src[i]
        if c == "\\":
            i += 2
            continue
        if c == "`":
            return i + 1
        if c == "$" and src[i + 1:i + 2] == "{":
            i = _skip_braced(src, i + 2)
            continue
        i += 1
    raise ValueError("unterminated template literal")


def _skip_braced(src, i):
    """Inside ${ ... }: return the index after the matching '}' (strings/templates/comments aware)."""
    depth = 1
    while i < len(src):
        c = src[i]
        if c in "'\"":
            i = _skip_string(src, i)
            continue
        if c == "`":
            i = _skip_template(src, i)
            continue
        if src.startswith("//", i):
            i = src.find("\n", i)
            i = len(src) if i < 0 else i
            continue
        if src.startswith("/*", i):
            i = src.index("*/", i) + 2
            continue
        if c == "{":
            depth += 1
        elif c == "}":
            depth -= 1
            if depth == 0:
                return i + 1
        i += 1
    raise ValueError("unterminated ${...}")


def _skip_regex(src, i):
    i += 1
    in_class = False
    while i < len(src):
        c = src[i]
        if c == "\\":
            i += 2
            continue
        if c == "\n":
            raise ValueError("unterminated regex")
        if c == "[":
            in_class = True
        elif c == "]":
            in_class = False
        elif c == "/" and not in_class:
            i += 1
            while i < len(src) and (src[i].isalnum() or src[i] == "_"):
                i += 1
            return i
        i += 1
    raise ValueError("unterminated regex")


PUNCT3 = ("...", "===", "!==", "**=", "<<=", ">>=", ">>>")
PUNCT2 = ("=>", "==", "!=", "<=", ">=", "&&", "||", "??", "?.", "++", "--", "+=", "-=", "*=", "/=",
          "%=", "&=", "|=", "^=", "**", "<<", ">>")


def tokenize(src):
    toks, i, n = [], 0, len(src)
    starts = [0]
    for m in re.finditer(r"\n", src):
        starts.append(m.end())

    def lineno(off):
        lo, hi = 0, len(starts)
        while lo < hi:
            mid = (lo + hi) // 2
            if starts[mid] <= off:
                lo = mid + 1
            else:
                hi = mid
        return lo
    while i < n:
        c = src[i]
        if c.isspace():
            i += 1
            continue
        if src.startswith("//", i):
            j = src.find("\n", i)
            i = n if j < 0 else j
            continue
        if src.startswith("/*", i):
            j = src.find("*/", i + 2)
            if j < 0:
                raise ValueError("unterminated comment at line %d" % lineno(i))
            i = j + 2
            continue
        start = i
        if c in "'\"":
            i = _skip_string(src, i)
            kind = "string"
        elif c == "`":
            i = _skip_template(src, i)
            kind = "template"
        elif c == "/":
            prev = toks[-1].text if toks else None
            if prev is None or (toks[-1].kind == "punct" and prev in REGEX_PREV) or \
                    (toks[-1].kind == "ident" and prev in REGEX_PREV):
                i = _skip_regex(src, i)
                kind = "regex"
            else:
                i += 2 if src.startswith("/=", i) else 1
                kind = "punct"
        elif c.isdigit() or (c == "." and i + 1 < n and src[i + 1].isdigit()):
            m = re.compile(r"(0[xXbBoO][0-9a-fA-F_]+|\d[\d_]*(\.\d*)?([eE][+-]?\d+)?|\.\d+([eE][+-]?\d+)?)n?").match(src, i)
            i = m.end()
            kind = "number"
        elif c.isalpha() or c in "_$":
            m = re.compile(r"[A-Za-z_$][\w$]*").match(src, i)
            i = m.end()
            kind = "ident"
        else:
            kind = "punct"
            for p in PUNCT3 + PUNCT2:
                if src.startswith(p, i):
                    i += len(p)
                    break
            else:
                i += 1
        toks.append(Tok(kind, src[start:i], start, i, lineno(start)))
    return toks


# --------------------------------------------------------------------------- discovery
OPEN, CLOSE = {"(": ")", "[": "]", "{": "}"}, {")", "]", "}"}


def expr_end(toks, k):
    """Index (exclusive) of the expression starting at token k: ends at ',' or ';' at depth 0, at a
    closing bracket that would go below depth 0, or at a new statement on a new line."""
    depth, j = 0, k
    while j < len(toks):
        t = toks[j]
        if t.kind == "punct" and t.text in OPEN:
            depth += 1
        elif t.kind == "punct" and t.text in CLOSE:
            if depth == 0:
                return j
            depth -= 1
        elif depth == 0 and t.kind == "punct" and t.text in (",", ";"):
            return j
        elif depth == 0 and j > k and t.kind == "ident" and t.text in STATEMENT_START \
                and t.line > toks[j - 1].line:
            return j
        j += 1
    return j


def const_defs(toks, src):
    """name -> list of (expr_text, line, start_tok, end_tok) for `const|let|var NAME = expr`."""
    defs = {}
    for k in range(len(toks) - 2):
        if toks[k].kind == "ident" and toks[k].text in ("const", "let", "var") and \
                toks[k + 1].kind == "ident" and toks[k + 2].text == "=":
            e = expr_end(toks, k + 3)
            if e > k + 3:
                text = src[toks[k + 3].start:toks[e - 1].end]
                defs.setdefault(toks[k + 1].text, []).append((text, toks[k + 1].line, k + 3, e))
    return defs


def _enclosing_object(toks, k):
    """Token index of the '{' that opens the object literal containing token k."""
    depth = 0
    for j in range(k - 1, -1, -1):
        t = toks[j]
        if t.kind == "punct" and t.text in CLOSE:
            depth += 1
        elif t.kind == "punct" and t.text in OPEN:
            if depth == 0:
                return j if t.text == "{" else None
            depth -= 1
    return None


def _static_prefix(tok):
    """Static prefix of a label/phase value: a string's content, a template's text before `${`."""
    if tok is None:
        return None
    if tok.kind == "string":
        return tok.text[1:-1]
    if tok.kind == "template":
        body = tok.text[1:-1]
        return body.split("${", 1)[0]
    return None


def call_sites(toks, src):
    """Every object-literal property named `schema` (incl. shorthand), with sibling label/phase."""
    sites = []
    for k, t in enumerate(toks):
        if t.kind != "ident" or t.text != "schema" or k == 0:
            continue
        prev, nxt = toks[k - 1], toks[k + 1] if k + 1 < len(toks) else None
        if not (prev.kind == "punct" and prev.text in ("{", ",")) or nxt is None:
            continue
        if nxt.text == ":":
            e = expr_end(toks, k + 2)
            expr = src[toks[k + 2].start:toks[e - 1].end] if e > k + 2 else ""
        elif nxt.text in (",", "}"):
            expr = "schema"                                  # shorthand: a variable named schema
        else:
            continue
        o = _enclosing_object(toks, k)
        label = phase = None
        if o is not None:
            depth, j = 0, o + 1
            while j < len(toks):
                tj = toks[j]
                if tj.kind == "punct" and tj.text in OPEN:
                    depth += 1
                elif tj.kind == "punct" and tj.text in CLOSE:
                    if depth == 0:
                        break
                    depth -= 1
                elif depth == 0 and tj.kind == "ident" and tj.text in ("label", "phase") and \
                        toks[j + 1].text == ":" and toks[j - 1].text in ("{", ","):
                    val = _static_prefix(toks[j + 2])
                    if tj.text == "label":
                        label = val
                    else:
                        phase = val
                j += 1
        if label is None:
            label_key = None
        else:
            label_key = re.split(r"[\s$]", label.strip(), 1)[0] or "<dynamic>"
        sites.append({"line": t.line, "expr": expr, "label": label, "label_key": label_key, "phase": phase})
    return sites


def references(expr):
    """Identifiers an expression reads (not property names, keys, or its own arrow params)."""
    toks = tokenize(expr)
    params = set()
    for k, t in enumerate(toks):                       # (a, b) => ... and function (a, b) { ... }
        if t.text == "=>":
            j = k - 1
            if j >= 0 and toks[j].kind == "ident":
                params.add(toks[j].text)
            elif j >= 0 and toks[j].text == ")":
                d = 0
                while j >= 0:
                    if toks[j].text == ")":
                        d += 1
                    elif toks[j].text == "(":
                        d -= 1
                        if d == 0:
                            break
                    elif toks[j].kind == "ident" and d == 1:
                        params.add(toks[j].text)
                    j -= 1
    refs = set()
    for k, t in enumerate(toks):
        if t.kind != "ident" or t.text in params or t.text in JS_GLOBALS:
            continue
        prev = toks[k - 1].text if k else None
        nxt = toks[k + 1].text if k + 1 < len(toks) else None
        if prev in (".", "?."):
            continue                                    # property access
        if nxt == ":" and prev in ("{", ","):
            continue                                    # object key
        if t.text in ("function", "return", "new", "typeof", "true", "false", "null"):
            continue
        refs.add(t.text)
    return refs


NODE_EVAL = r"""
const vm = require('vm');
const job = JSON.parse(require('fs').readFileSync(0, 'utf8'));
const ctx = vm.createContext(Object.create(null));
const out = { defs: {}, values: {} };
for (const d of job.order) {
  try { vm.runInContext('var ' + d.name + ' = (' + d.expr + ');', ctx, { timeout: 1000 }); out.defs[d.name] = 'ok'; }
  catch (e) { out.defs[d.name] = 'error: ' + e.message; }
}
for (const t of job.targets) {
  try {
    const v = vm.runInContext('(' + t.expr + ')', ctx, { timeout: 1000 });
    out.values[t.id] = (v && typeof v === 'object') ? JSON.parse(JSON.stringify(v)) : { __not_a_schema__: typeof v };
  } catch (e) { out.values[t.id] = { __error__: e.message }; }
}
process.stdout.write(JSON.stringify(out));
"""


def resolve(targets, defs):
    """Evaluate each target expression with the const closure it needs. Returns id -> schema or error."""
    errors, order, seen = {}, [], set()

    def need(name, stack):
        if name in seen:
            return True
        if name in stack:
            raise ValueError("cyclic reference via %s" % name)
        if name not in defs:
            raise ValueError("unresolved reference %r" % name)
        variants = {d[0] for d in defs[name]}
        if len(variants) > 1:
            raise ValueError("ambiguous: %r is defined %d different ways" % (name, len(variants)))
        expr = defs[name][0][0]
        for r in sorted(references(expr)):
            need(r, stack | {name})
        seen.add(name)
        order.append({"name": name, "expr": expr})
        return True
    ready = []
    for t in targets:
        try:
            for r in sorted(references(t["expr"])):
                need(r, frozenset())
            ready.append(t)
        except ValueError as exc:
            errors[t["id"]] = str(exc)
    proc = subprocess.run(["node", "-e", NODE_EVAL], input=json.dumps({"order": order, "targets": ready}),
                          capture_output=True, text=True, timeout=120)
    if proc.returncode != 0:
        raise RuntimeError("node evaluation failed: %s" % proc.stderr.strip()[:500])
    out = json.loads(proc.stdout)
    bad_defs = {k: v for k, v in out["defs"].items() if v != "ok"}
    for tid, val in out["values"].items():
        if "__error__" in val:
            errors[tid] = "evaluation error: %s%s" % (val["__error__"], (" (defs: %s)" % bad_defs) if bad_defs else "")
        elif "__not_a_schema__" in val:
            errors[tid] = "not a schema object (%s)" % val["__not_a_schema__"]
    values = {tid: v for tid, v in out["values"].items() if tid not in errors}
    return values, errors


def field_paths(schema, path=""):
    """Leaves, array nodes and open-object boundaries of a resolved JSON schema, in order."""
    out = []
    t = schema.get("type")
    types = t if isinstance(t, list) else [t]
    if "object" in types or "properties" in schema:
        for k, v in (schema.get("properties") or {}).items():
            out += field_paths(v if isinstance(v, dict) else {}, "%s.%s" % (path, k) if path else k)
        if schema.get("additionalProperties") is True or (
                "additionalProperties" not in schema and not schema.get("properties")):
            out.append({"path": (path + ".*") if path else "*", "kind": "open_object"})
        return out
    if "array" in types:
        items = schema.get("items") if isinstance(schema.get("items"), dict) else {}
        out.append({"path": path + "[]", "kind": "array"})
        it = items.get("type")
        if it == "object" or "properties" in items:
            out += field_paths(items, path + "[]")
        return out
    leaf = {"path": path, "kind": "leaf", "type": t}
    if "enum" in schema:
        leaf["enum"] = schema["enum"]
    return [leaf]


def discover(root=GEAK_ROOT, files=ENTRY_FILES):
    """{file: {"sites": [...], "roots": {root_id: {"paths": [...], "sites": [...]}}, "errors": [...]}}"""
    result = {}
    for rel in files:
        src = open(os.path.join(root, rel), encoding="utf-8").read()
        toks = tokenize(src)
        defs = const_defs(toks, src)
        sites = call_sites(toks, src)
        targets, errors, ignored = [], [], []
        for s in sites:
            expr = s["expr"].strip()
            if re.fullmatch(r"'[^'\n]*'|\"[^\"\n]*\"", expr):
                ignored.append("line %d: schema: %s is a string tag, not an agent schema" % (s["line"], expr))
                continue                                 # e.g. LLM_TL = { schema: 'geak.agent_timeline/1' }
            if re.fullmatch(r"[A-Za-z_$][\w$]*", expr):
                s["root"] = expr
            else:
                if not (s["label_key"] or s["phase"]):
                    errors.append("line %d: inline schema without a static label or phase" % s["line"])
                    continue
                s["root"] = "inline:%s@%s" % (s["label_key"] or "", s["phase"] or "")
            targets.append({"id": s["root"], "expr": expr})
        uniq = {}
        for t in targets:
            if t["id"] in uniq and uniq[t["id"]] != t["expr"]:
                errors.append("root %s is bound to two different schema expressions" % t["id"])
            uniq[t["id"]] = t["expr"]
        values, rerr = resolve([{"id": k, "expr": v} for k, v in uniq.items()], defs)
        errors += ["%s: %s" % (k, v) for k, v in sorted(rerr.items())]
        roots = {}
        for rid, schema in values.items():
            roots[rid] = {"paths": field_paths(schema),
                          "sites": sorted({(s["label_key"] or "", s["phase"] or "") for s in sites
                                           if s.get("root") == rid})}
        result[rel] = {"roots": roots, "errors": errors, "ignored": ignored, "site_count": len(sites)}
    return result


# --------------------------------------------------------------------------- reconcile
def lint(registry, found, root=GEAK_ROOT, files=ENTRY_FILES):
    """Problems (empty list = pass). Reconciles in both directions, per file and root."""
    probs = []
    decisions = {d["id"]: d for d in registry.get("decisions", [])}
    for d in decisions.values():
        if d.get("capture") not in ("not_implemented", "implemented"):
            probs.append("decision %s: capture must be not_implemented or implemented" % d["id"])
        if d.get("capture") == "implemented":
            marker = "EIKOS-CAPTURE:%s" % d["id"]
            if not any(marker in open(os.path.join(root, f), encoding="utf-8").read() for f in files):
                probs.append("decision %s claims capture but no %s marker exists in the sources" % (d["id"], marker))
        for f in d.get("internal_files", []):
            if not os.path.exists(os.path.join(root, f)):
                probs.append("decision %s: internal site file %s is missing" % (d["id"], f))
    reg_files = registry.get("files", {})
    for rel, info in found.items():
        probs += ["%s: %s" % (rel, e) for e in info["errors"]]
        reg_roots = reg_files.get(rel, {})
        for rid, r in info["roots"].items():
            reg = reg_roots.get(rid)
            if reg is None:
                probs.append("%s: schema root %s (call sites %s) is not in the registry" % (rel, rid, r["sites"]))
                continue
            reg_sites = sorted({tuple(s) for s in reg.get("call_sites", [])})
            if reg_sites != r["sites"]:
                probs.append("%s: %s call sites differ: found %s, registered %s" % (rel, rid, r["sites"], reg_sites))
            fields = reg.get("fields", {})
            found_paths = {p["path"] for p in r["paths"]}
            for p in r["paths"]:
                entry = fields.get(p["path"])
                if entry is None:
                    probs.append("%s: %s.%s (%s%s) is not classified" % (
                        rel, rid, p["path"], p["kind"], (" " + str(p.get("type"))) if p.get("type") else ""))
                    continue
                cls = entry.get("class")
                if cls not in CLASSES:
                    probs.append("%s: %s.%s has unknown class %r" % (rel, rid, p["path"], cls))
                if cls in NEEDS_DECISION and entry.get("decision") not in decisions:
                    probs.append("%s: %s.%s is %s but names no registered decision" % (rel, rid, p["path"], cls))
                if cls == "derived_fact" and not entry.get("policy"):
                    probs.append("%s: %s.%s is derived_fact without a policy citation" % (rel, rid, p["path"]))
                if p["kind"] == "open_object" and cls != "open_object":
                    probs.append("%s: %s.%s is an open-object boundary; class must be open_object" % (rel, rid, p["path"]))
            for path in fields:
                if path not in found_paths:
                    probs.append("%s: registered %s.%s no longer exists in the source" % (rel, rid, path))
        for rid in reg_roots:
            if rid not in info["roots"]:
                probs.append("%s: registered schema root %s no longer has a call site" % (rel, rid))
    for rel in reg_files:
        if rel not in found:
            probs.append("registry names %s, which is not a linted entry file" % rel)
    return probs


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--discover", action="store_true")
    ap.add_argument("--registry", default=REGISTRY)
    a = ap.parse_args(argv)
    found = discover()
    if a.discover:
        print(json.dumps(found, indent=1))
        return 0
    with open(a.registry, encoding="utf-8") as fh:
        registry = json.load(fh)
    probs = lint(registry, found)
    for p in probs:
        print("FAIL " + p)
    roots = sum(len(v["roots"]) for v in found.values())
    paths = sum(len(r["paths"]) for v in found.values() for r in v["roots"].values())
    print("%s: %d files, %d schema roots, %d field paths, %d problem(s)" % (
        "PASS" if not probs else "FAIL", len(found), roots, paths, len(probs)))
    return 0 if not probs else 1


if __name__ == "__main__":
    sys.exit(main())
