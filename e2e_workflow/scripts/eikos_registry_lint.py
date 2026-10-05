#!/usr/bin/env python3
"""Lint: every typed field an agent returns in a GEAK workflow is classified in the decision registry.

The registry (eikos_questions/decision_registry.json) is the required declaration for the "consult
Eikos on judgment decisions" rule (design: research/eikos_hard_rule_proposal_20261005.md, rev 2).
This lint does NOT trust the registry to say what exists. It discovers, independently:

  * every agent call site that passes a `schema` -- key written as `schema`, `'schema'`,
    `"schema"` or `['schema']`, or the `{ schema }` shorthand -- including call sites inside the
    executable `${...}` parts of template literals (literal template text is never code);
  * the value of each schema expression, by evaluating it in an empty `node` vm context together
    with the `const` definitions it references;
  * every field path in the resolved schema: leaves, array nodes (at any nesting depth) and every
    open-object boundary. JSON Schema leaves `additionalProperties` open unless it is `false`; a
    schema-valued `additionalProperties` is an open boundary whose value schema is walked too.

Supported boundary -- anything outside it is an ERROR, never a silent pass:
  * a call-site schema is a literal, or a name whose closure consists only of `const` bindings, each
    bound exactly once in the file and used nowhere except as a schema value or inside another
    such initializer (so no shadowing parameter, no `let`/`var`, no member access or mutation, no
    escape into other code);
  * schema keywords are limited to SCHEMA_KEYWORDS (no oneOf/anyOf/allOf/$ref/not/if/
    patternProperties/tuple items ...);
  * a call site is identified by its full label (template parts normalised to `${}`) and phase,
    and counted with multiplicity, so a second call with the same label is a new site.
Not covered: computed keys built from non-literal expressions, options objects assembled at run
time, and judgments made in free text or inside an agent's own reasoning.

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
SCHEMA_KEYWORDS = {"type", "properties", "required", "additionalProperties", "items", "enum",
                   "description", "minimum", "maximum", "exclusiveMinimum", "exclusiveMaximum",
                   "minItems", "maxItems", "minLength", "maxLength", "pattern", "format", "default",
                   "title", "minProperties", "maxProperties", "uniqueItems"}
JS_GLOBALS = {"Object", "JSON", "Array", "String", "Number", "Boolean", "Math", "undefined", "null",
              "true", "false", "Infinity", "NaN"}
REGEX_PREV = set("(,=:[!&|?{};+-*%<>~^") | {"return", "typeof", "instanceof", "in", "of", "new",
                                           "delete", "void", "throw", "case", "do", "else", "=>", "${"}
STATEMENT_START = {"const", "let", "var", "function", "if", "for", "while", "return", "export",
                   "class", "async", "await", "try", "throw", "log", "phase", "do", "switch"}
KEYWORDS = {"if", "for", "while", "switch", "catch", "function", "return", "typeof", "new", "await",
            "async", "of", "in", "else", "do", "try", "throw", "case", "default", "delete", "void",
            "instanceof", "this", "super", "class", "const", "let", "var", "yield", "import", "export"}


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
        if c == "\n":
            raise ValueError("unterminated string")
        i += 1
    raise ValueError("unterminated string")


def _skip_braced(src, i):
    """i just after `${`: index after the matching `}` (strings/templates/comments aware)."""
    depth = 1
    while i < len(src):
        c = src[i]
        if c in "'\"":
            i = _skip_string(src, i)
            continue
        if c == "`":
            i = _template_parts(src, i)[0]
            continue
        if src.startswith("//", i):
            j = src.find("\n", i)
            i = len(src) if j < 0 else j
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


def _template_parts(src, i):
    """i at the opening backtick. Returns (end, [(body_start, body_end), ...]) for its ${} bodies."""
    i += 1
    parts = []
    while i < len(src):
        c = src[i]
        if c == "\\":
            i += 2
            continue
        if c == "`":
            return i + 1, parts
        if c == "$" and src[i + 1:i + 2] == "{":
            end = _skip_braced(src, i + 2)
            parts.append((i + 2, end - 1))
            i = end
            continue
        i += 1
    raise ValueError("unterminated template literal")


ESCAPES = {"n": "\n", "t": "\t", "r": "\r", "b": "\b", "f": "\f", "v": "\v", "0": "\0"}


def js_string_value(text):
    """The value of a JS string literal token (quotes included), escapes decoded, so a key's identity
    never depends on its spelling ('sch\\u0065ma' is 'schema'). Raises ValueError if undecodable."""
    body, out, i = text[1:-1], [], 0
    while i < len(body):
        c = body[i]
        if c != "\\":
            out.append(c)
            i += 1
            continue
        n = body[i + 1] if i + 1 < len(body) else ""
        if n in ESCAPES and not (n == "0" and body[i + 2:i + 3].isdigit()):
            out.append(ESCAPES[n])
            i += 2
        elif n == "x":
            out.append(chr(int(body[i + 2:i + 4], 16)))
            i += 4
        elif n == "u" and body[i + 2:i + 3] == "{":
            j = body.index("}", i)
            out.append(chr(int(body[i + 3:j], 16)))
            i = j + 1
        elif n == "u":
            out.append(chr(int(body[i + 2:i + 6], 16)))
            i += 6
        elif n in "\r\n\u2028\u2029":
            i += 3 if body[i + 1:i + 3] == "\r\n" else 2       # line continuation
        elif n.isdigit():
            raise ValueError("legacy octal escape in %s" % text)
        else:
            out.append(n)
            i += 2
    return "".join(out)


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
NUM_RE = re.compile(r"(0[xXbBoO][0-9a-fA-F_]+|\d[\d_]*(\.\d*)?([eE][+-]?\d+)?|\.\d+([eE][+-]?\d+)?)n?")
IDENT_RE = re.compile(r"[A-Za-z_$][\w$]*")


def tokenize(src, lo=0, hi=None, starts=None):
    """Tokens of src[lo:hi]. A template literal yields one `template` token followed, for each
    `${...}`, by a synthetic `${` token, the body's own tokens and a synthetic `}` token, so code
    inside interpolations is visible and bracket depth stays balanced."""
    hi = len(src) if hi is None else hi
    if starts is None:
        starts = [0] + [m.end() for m in re.finditer(r"\n", src)]

    def lineno(off):
        a, b = 0, len(starts)
        while a < b:
            mid = (a + b) // 2
            if starts[mid] <= off:
                a = mid + 1
            else:
                b = mid
        return a
    toks, i = [], lo
    while i < hi:
        c = src[i]
        if c.isspace():
            i += 1
            continue
        if src.startswith("//", i):
            j = src.find("\n", i)
            i = hi if j < 0 or j > hi else j
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
            toks.append(Tok("string", src[start:i], start, i, lineno(start)))
            continue
        if c == "`":
            i, parts = _template_parts(src, i)
            toks.append(Tok("template", src[start:i], start, i, lineno(start)))
            for b0, b1 in parts:
                toks.append(Tok("punct", "${", b0 - 2, b0, lineno(b0)))
                toks += tokenize(src, b0, b1, starts)
                toks.append(Tok("punct", "}", b1, b1 + 1, lineno(b1)))
            continue
        if c == "/":
            prev = toks[-1].text if toks else None
            if prev is None or prev in REGEX_PREV:
                i = _skip_regex(src, i)
                toks.append(Tok("regex", src[start:i], start, i, lineno(start)))
                continue
            i += 2 if src.startswith("/=", i) else 1
            toks.append(Tok("punct", src[start:i], start, i, lineno(start)))
            continue
        if c.isdigit() or (c == "." and i + 1 < hi and src[i + 1].isdigit()):
            i = NUM_RE.match(src, i).end()
            kind = "number"
        elif c.isalpha() or c in "_$":
            i = IDENT_RE.match(src, i).end()
            kind = "ident"
        elif c == "\\":
            raise ValueError("unicode-escaped identifier at line %d is not supported" % lineno(i))
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
OPEN, CLOSE = {"(": ")", "[": "]", "{": "}", "${": "}"}, {")", "]", "}"}


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


def _span(src, toks, a, e):
    """Source text of tokens a..e-1. Ends at the furthest token end: a template token spans its whole
    literal, while the synthetic tokens of its `${}` bodies that follow it end earlier."""
    return src[toks[a].start:max(t.end for t in toks[a:e])]


def _matching(toks, k):
    """Index of the bracket closing the one opened at token k."""
    depth = 0
    for j in range(k, len(toks)):
        if toks[j].kind == "punct" and toks[j].text in OPEN:
            depth += 1
        elif toks[j].kind == "punct" and toks[j].text in CLOSE:
            depth -= 1
            if depth == 0:
                return j
    return len(toks) - 1


def const_defs(toks, src):
    """name -> list of (expr_text, line, expr_start_tok, expr_end_tok) for `const NAME = expr`."""
    defs = {}
    for k in range(len(toks) - 2):
        if toks[k].text == "const" and toks[k + 1].kind == "ident" and toks[k + 2].text == "=":
            e = expr_end(toks, k + 3)
            if e > k + 3:
                defs.setdefault(toks[k + 1].text, []).append((_span(src, toks, k + 3, e),
                                                              toks[k + 1].line, k + 3, e))
    return defs


def bindings(toks):
    """name -> list of (kind, line) for every binding form the lint can see: const/let/var (incl.
    destructuring), function and class names, function/arrow/method parameters, catch parameters."""
    out = {}

    def add(name, kind, line):
        out.setdefault(name, []).append((kind, line))

    def bound_in(a, b, kind):
        depth, after_eq = 0, False
        for j in range(a, b + 1):
            t = toks[j]
            if t.kind == "punct" and t.text in OPEN:
                depth += 1
            elif t.kind == "punct" and t.text in CLOSE:
                depth -= 1
            if t.kind == "punct" and t.text == "=":
                after_eq = True
            elif t.text == "," and depth <= 1:
                after_eq = False
            elif t.kind == "ident" and not after_eq and t.text not in KEYWORDS:
                if not (j + 1 < len(toks) and toks[j + 1].text == ":"):
                    add(t.text, kind, t.line)            # an object-pattern key is not a binding
    for k, t in enumerate(toks):
        if t.text in ("const", "let", "var") and t.kind == "ident" and k + 1 < len(toks):
            n = toks[k + 1]
            if n.kind == "ident":
                add(n.text, t.text, n.line)
            elif n.text in ("{", "["):
                bound_in(k + 1, _matching(toks, k + 1), t.text + "-pattern")
        elif t.text in ("function", "class") and t.kind == "ident" and k + 1 < len(toks):
            j = k + 1
            if toks[j].text == "*":
                j += 1
            if toks[j].kind == "ident":
                add(toks[j].text, t.text, toks[j].line)
                j += 1
            if t.text == "function" and j < len(toks) and toks[j].text == "(":
                bound_in(j, _matching(toks, j), "param")
        elif t.text == "catch" and k + 1 < len(toks) and toks[k + 1].text == "(":
            bound_in(k + 1, _matching(toks, k + 1), "catch")
        elif t.text == "=>" and k > 0:
            p = toks[k - 1]
            if p.kind == "ident":
                add(p.text, "param", p.line)
            elif p.text == ")":
                d, j = 0, k - 1
                while j >= 0:
                    if toks[j].text == ")":
                        d += 1
                    elif toks[j].text == "(":
                        d -= 1
                        if d == 0:
                            break
                    j -= 1
                bound_in(j, k - 1, "param")
        elif t.kind == "ident" and t.text not in KEYWORDS and k + 1 < len(toks) and toks[k + 1].text == "(" \
                and k > 0 and toks[k - 1].text in ("{", ",", "}", ";", "async", "static", "get", "set"):
            close = _matching(toks, k + 1)             # method shorthand: name(params) { ... }
            if close + 1 < len(toks) and toks[close + 1].text == "{":
                bound_in(k + 1, close, "param")
    return out


def _enclosing_object(toks, k):
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


def _normalize_template(text):
    """`a ${x} b` -> 'a ${} b', so a label's text is a stable identity."""
    end, parts = _template_parts(text, 0)
    out, last = [], 1
    for b0, b1 in parts:
        out.append(text[last:b0 - 2])
        out.append("${}")
        last = b1 + 1
    out.append(text[last:end - 1])
    return "".join(out)


def _label_value(tok):
    if tok is None:
        return None
    if tok.kind == "string":
        return js_string_value(tok.text)
    if tok.kind == "template":
        return _normalize_template(tok.text)
    return None


def _is_key(toks, k, name):
    """Token k starts an object-literal key spelled `name`, 'name', "name" or ['name'].
    Returns (index of the token before the value, shorthand?) or None."""
    if k == 0 or toks[k - 1].text not in ("{", ","):
        return None
    t = toks[k]
    if t.kind == "ident" and t.text == name:
        nxt = toks[k + 1].text if k + 1 < len(toks) else None
        if nxt == ":":
            return k + 1, False
        if nxt in (",", "}"):
            return k, True
        return None
    if t.kind == "string" and k + 1 < len(toks) and toks[k + 1].text == ":" and js_string_value(t.text) == name:
        return k + 1, False
    if t.text == "[" and k + 3 < len(toks) and toks[k + 1].kind == "string" and toks[k + 2].text == "]" \
            and toks[k + 3].text == ":" and js_string_value(toks[k + 1].text) == name:
        return k + 3, False
    return None


def call_sites(toks, src):
    """Every object-literal property keyed `schema` (any static spelling), with sibling label/phase."""
    sites = []
    for k in range(1, len(toks)):
        hit = _is_key(toks, k, "schema")
        if not hit:
            continue
        colon, shorthand = hit
        if shorthand:
            expr, rng = "schema", (k, k + 1)
        else:
            e = expr_end(toks, colon + 1)
            expr = _span(src, toks, colon + 1, e) if e > colon + 1 else ""
            rng = (colon + 1, e)
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
                elif depth == 0:
                    for key in ("label", "phase"):
                        h = _is_key(toks, j, key)
                        if h and not h[1]:
                            val = _label_value(toks[h[0] + 1])
                            if key == "label":
                                label = val
                            else:
                                phase = val
                j += 1
        label_key = None if label is None else (re.split(r"[\s$]", label.strip(), 1)[0] or "<dynamic>")
        sites.append({"line": toks[k].line, "expr": expr, "expr_range": rng, "label": label,
                      "label_key": label_key, "phase": phase})
    return sites


def references(expr):
    """Identifiers an expression reads (not property names, keys, or its own arrow params)."""
    toks = tokenize(expr)
    params = set()
    for k, t in enumerate(toks):
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
            continue
        if nxt == ":" and prev in ("{", ","):
            continue
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


def closure(expr, defs):
    """Const names (dependency order) a schema expression needs. Raises ValueError if unresolved."""
    order, seen = [], set()

    def need(name, stack):
        if name in seen:
            return
        if name in stack:
            raise ValueError("cyclic reference via %s" % name)
        if name not in defs:
            raise ValueError("unresolved reference %r (not a const in this file)" % name)
        if len(defs[name]) > 1:
            raise ValueError("%r is declared %d times; its binding at the call site is ambiguous"
                             % (name, len(defs[name])))
        for r in sorted(references(defs[name][0][0])):
            need(r, stack | {name})
        seen.add(name)
        order.append(name)
    for r in sorted(references(expr)):
        need(r, frozenset())
    return order


ASSIGN_OPS = {"=", "+=", "-=", "*=", "/=", "%=", "**=", "<<=", ">>=", ">>>=", "&=", "|=", "^=",
              "&&=", "||=", "??=", "++", "--"}


def construction_problem(expr, callable_names):
    """'' if expr only BUILDS a value: literals, aliases, arrow helpers, and calls to the closure's own
    helper names. Any assignment, increment, delete, new, member access, or call to anything else
    (Object.assign, JSON.parse, a method ...) could mutate or depend on run-time state."""
    toks = tokenize(expr)
    for k, t in enumerate(toks):
        if t.kind == "punct" and t.text in ASSIGN_OPS:
            return "assignment %r" % t.text
        if t.kind == "ident" and t.text in ("delete", "new", "await", "yield", "this"):
            return "%r" % t.text
        if t.kind == "punct" and t.text in (".", "?."):
            return "member access"
        if t.kind == "punct" and t.text == "[" and k and (toks[k - 1].kind == "ident" or toks[k - 1].text in (")", "]")):
            return "computed member access"
        if t.kind == "punct" and t.text == "(" and k:
            callee = toks[k - 1]
            if callee.kind == "ident" and callee.text not in callable_names and callee.text not in KEYWORDS:
                return "call to %r" % callee.text
            if callee.text in (")", "]"):
                return "call of a computed value"
    return ""


def check_closure_use(names, defs, binds, toks, sites):
    """Supported boundary for names a schema depends on: exactly one binding, a `const`, and no use
    outside schema expressions and the initializers of other closure names. Returns problems."""
    probs = []
    allowed = [defs[n][0][2:4] for n in names] + [s["expr_range"] for s in sites]
    for name in names:
        kinds = binds.get(name, [])
        if len(kinds) != 1 or kinds[0][0] != "const":
            probs.append("%r is rebound or shadowed (%s); a call site may not see its const initializer"
                         % (name, ", ".join("%s@%d" % x for x in kinds)))
        decl = defs[name][0][2] - 2                      # the NAME token of `const NAME =`
        for k, t in enumerate(toks):
            if t.kind != "ident" or t.text != name or k == decl:
                continue
            if k and toks[k - 1].text in (".", "?."):
                continue                                 # someone else's property of the same name
            if k + 1 < len(toks) and toks[k + 1].text == ":" and k and toks[k - 1].text in ("{", ","):
                continue                                 # an object key of the same name
            if not any(a <= k < b for a, b in allowed):
                probs.append("%r is used outside a schema expression at line %d (possible mutation or escape)"
                             % (name, t.line))
                break
    return probs


def resolve(targets, defs):
    """Evaluate each target expression with its const closure. Returns (id -> schema, id -> error)."""
    errors, order, seen, ready = {}, [], set(), []
    for t in targets:
        try:
            for name in closure(t["expr"], defs):
                if name not in seen:
                    seen.add(name)
                    order.append({"name": name, "expr": defs[name][0][0]})
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
    return {tid: v for tid, v in out["values"].items() if tid not in errors}, errors


def _structure(schema, where):
    """'object', 'array', 'scalar' or 'unconstrained'. A schema that is both (type ['object','array'],
    or object keywords next to items) is a structural union: rejected, never half-walked. A schema with
    no type, no structural keyword and no enum (e.g. `{}`) admits ANY value, objects and arrays
    included: it is 'unconstrained', an open boundary, never a scalar."""
    t = schema.get("type")
    if isinstance(t, list) and not t:
        raise ValueError("%s: empty type list" % where)
    types = set(t if isinstance(t, list) else [t]) - {None}
    is_obj = "object" in types or "properties" in schema or "additionalProperties" in schema
    is_arr = "array" in types or "items" in schema
    if is_obj and is_arr:
        raise ValueError("%s: structural union (object and array) is not supported" % where)
    if types and is_obj and "object" not in types:
        raise ValueError("%s: object keywords on a schema typed %s" % (where, sorted(types)))
    if types and is_arr and "array" not in types:
        raise ValueError("%s: items on a schema typed %s" % (where, sorted(types)))
    if is_obj:
        return "object"
    if is_arr:
        return "array"
    if types:
        return "scalar"
    if "enum" in schema:
        if any(isinstance(v, (dict, list)) for v in schema["enum"]):
            raise ValueError("%s: enum of objects/arrays is not supported" % where)
        return "scalar"
    return "unconstrained"


def field_paths(schema, path=""):
    """Leaves, array nodes and open-object boundaries of a resolved JSON schema. Raises ValueError
    on vocabulary outside SCHEMA_KEYWORDS."""
    where = path or "<root>"
    if not isinstance(schema, dict):
        raise ValueError("%s: schema is not an object" % where)
    extra = sorted(set(schema) - SCHEMA_KEYWORDS)
    if extra:
        raise ValueError("%s: unsupported schema keyword(s) %s" % (where, ", ".join(extra)))
    t = schema.get("type")
    types = t if isinstance(t, list) else [t]
    shape = _structure(schema, where)
    out = []
    if shape == "object":
        for k, v in (schema.get("properties") or {}).items():
            out += field_paths(v, "%s.%s" % (path, k) if path else k)
        ap = schema.get("additionalProperties", True)  # JSON Schema: absent means allowed
        star = (path + ".*") if path else "*"
        if ap is True:
            out.append({"path": star, "kind": "open_object"})
        elif isinstance(ap, dict):
            out.append({"path": star, "kind": "open_object", "values": "schema"})
            out += field_paths(ap, (path + ".<*>") if path else "<*>")
        elif ap is not False:
            raise ValueError("%s: unsupported additionalProperties value %r" % (where, ap))
        return out
    if shape == "array":
        node = {"path": path + "[]", "kind": "array"}
        out.append(node)
        items = schema.get("items")
        if items is None:
            out.append({"path": path + "[].*", "kind": "open_object"})
        elif not isinstance(items, dict):
            raise ValueError("%s: tuple-form items are not supported" % where)
        elif _structure(items, where + "[]") == "unconstrained":
            out.append({"path": path + "[].*", "kind": "open_object", "values": "any"})
        elif _structure(items, where + "[]") == "scalar":    # scalar items: one entry, not two
            sub = field_paths(items, path + "[]")
            node["items_type"] = sub[0]["type"]
            if "enum" in sub[0]:
                node["enum"] = sub[0]["enum"]
        else:
            out += field_paths(items, path + "[]")
        return out
    if shape == "unconstrained":                       # `{}`: any value -> an open boundary
        return [{"path": path or "*", "kind": "open_object", "values": "any"}]
    leaf = {"path": path, "kind": "leaf", "type": t}
    if "enum" in schema:
        leaf["enum"] = schema["enum"]
    return [leaf]


def discover(root=GEAK_ROOT, files=ENTRY_FILES):
    """{file: {"roots": {root_id: {"paths": [...], "sites": [[label, phase], ...]}}, "errors": [...]}}"""
    result = {}
    for rel in files:
        src = open(os.path.join(root, rel), encoding="utf-8").read()
        try:
            toks = tokenize(src)
        except ValueError as exc:
            result[rel] = {"roots": {}, "errors": ["tokenize: %s" % exc], "ignored": [],
                           "unused_schemas": [], "site_count": 0}
            continue
        defs = const_defs(toks, src)
        sites = call_sites(toks, src)
        targets, errors, ignored, used = [], [], [], []
        for s in sites:
            expr = s["expr"].strip()
            if re.fullmatch(r"'[^'\n]*'|\"[^\"\n]*\"", expr):
                ignored.append("line %d: schema: %s is a string tag, not an agent schema" % (s["line"], expr))
                continue                                 # e.g. LLM_TL = { schema: 'geak.agent_timeline/1' }
            if re.fullmatch(r"[A-Za-z_$][\w$]*", expr):
                s["root"] = expr
            elif s["label_key"] or s["phase"]:
                s["root"] = "inline:%s@%s" % (s["label_key"] or "", s["phase"] or "")
            else:
                errors.append("line %d: inline schema without a static label or phase" % s["line"])
                continue
            used.append(s)
            targets.append({"id": s["root"], "expr": expr})
        uniq = {}
        for t in targets:
            if t["id"] in uniq and uniq[t["id"]] != t["expr"]:
                errors.append("root %s is bound to two different schema expressions" % t["id"])
            uniq[t["id"]] = t["expr"]
        names = set()
        for expr in uniq.values():
            try:
                names.update(closure(expr, defs))
            except ValueError:
                pass                                     # reported by resolve() below
        # Other consts built only from those names (e.g. a schema no call site uses yet) may use them
        # too; they get the same rules and are listed, so an unused schema is visible, not hidden.
        pure = set(names)
        while True:
            more = {n for n, d in defs.items() if n not in pure and len(d) == 1 and
                    references(d[0][0]) and references(d[0][0]) <= pure
                    and not construction_problem(d[0][0], pure | {n})}
            if not more:
                break
            pure |= more
        unused = sorted(pure - names)
        for n in sorted(pure):
            why = construction_problem(defs[n][0][0], pure)
            if why:
                errors.append("%r is not built by construction only (%s); the lint cannot see what it "
                              "does to the schemas it touches" % (n, why))
        errors += check_closure_use(sorted(pure), defs, bindings(toks), toks, used)
        values, rerr = resolve([{"id": k, "expr": v} for k, v in uniq.items()], defs)
        errors += ["%s: %s" % (k, v) for k, v in sorted(rerr.items())]
        roots = {}
        for rid, schema in values.items():
            try:
                paths = field_paths(schema)
            except ValueError as exc:
                errors.append("%s: %s" % (rid, exc))
                continue
            roots[rid] = {"paths": paths,
                          "sites": sorted([s["label"] or "", s["phase"] or ""] for s in used if s["root"] == rid)}
        result[rel] = {"roots": roots, "errors": errors, "ignored": ignored, "unused_schemas": unused,
                       "site_count": len(sites)}
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
            reg_sites = sorted([list(s) for s in reg.get("call_sites", [])])
            if reg_sites != r["sites"]:
                probs.append("%s: %s call sites differ (each syntactic site counts): found %s, registered %s"
                             % (rel, rid, r["sites"], reg_sites))
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
    for rel, v in found.items():
        for n in v.get("unused_schemas", []):
            print("NOTE %s: schema definition %s has no call site (not a decision site until one exists)" % (rel, n))
    print("%s: %d files, %d schema roots, %d field paths, %d problem(s)" % (
        "PASS" if not probs else "FAIL", len(found), roots, paths, len(probs)))
    return 0 if not probs else 1


if __name__ == "__main__":
    sys.exit(main())
