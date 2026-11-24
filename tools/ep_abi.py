# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
"""Frozen-contract checks for GEAK ONNX-Runtime EP kernel tasks.

Copied next to the task runner (scripts/ep_abi.py) by tools/gen_ep_task.py.
The generator records a manifest of the pristine kernel; the runner's
`compile` compares every candidate against it before the EP build and checks
the rebuilt kernel library's exports after it.

The manifest freezes:
  extern_c        every `extern "C"` function the kernel file defines, by its
                  formatting-normalized signature. The candidate is compiled
                  inside the EP tree, whose ABI header (hip_custom_kernels.h)
                  is never overlaid, so a definition that disagrees with the
                  header is also a compile error as long as the header stays
                  included (required_includes).
  functions       host dispatch gates, by a hash of their formatting-normalized
                  definitions (comments and whitespace do not count).
  constants       namespace-scope constants: the listed ones, plus every one a
                  frozen function reads.
  exports         unmangled symbols the kernel library exports (optional).
"""
import functools
import hashlib
import re
import subprocess

_IDENT = re.compile(r"[A-Za-z_]\w*")


@functools.lru_cache(maxsize=4)
def strip_comments(src):
    """Return (code, mask): `code` is `src` with comments replaced by spaces;
    `mask` is `code` with string/char literal contents also blanked and every
    preprocessor line (continuation lines too) reduced to a lone `#`, so brace
    and paren matching on `mask` never sees a brace inside a literal or a macro
    body. Both keep `src`'s offsets."""
    code, mask = list(src), list(src)
    i, n = 0, len(src)
    while i < n:
        c = src[i]
        if src.startswith("//", i):
            j = src.find("\n", i)
            j = n if j < 0 else j
            for k in range(i, j):
                code[k] = mask[k] = " "
            i = j
        elif src.startswith("/*", i):
            j = src.find("*/", i + 2)
            j = n if j < 0 else j + 2
            for k in range(i, j):
                if src[k] != "\n":
                    code[k] = mask[k] = " "
            i = j
        elif c == "R" and src.startswith('R"', i) and (i == 0 or not src[i - 1].isalnum()):
            open_paren = src.find("(", i + 2)
            delim = src[i + 2:open_paren]
            j = src.find(")" + delim + '"', open_paren)
            j = n if j < 0 else j + len(delim) + 2
            for k in range(open_paren + 1, j - len(delim) - 2):
                mask[k] = " "
            i = j
        elif c in "\"'":
            j = i + 1
            while j < n and src[j] != c and src[j] != "\n":
                j += 2 if src[j] == "\\" else 1
            for k in range(i + 1, min(j, n)):
                mask[k] = " "
            i = j + 1
        else:
            i += 1
    start, directive = 0, False
    for line in "".join(code).split("\n"):
        if line and (directive or line.lstrip().startswith("#")):
            mask[start:start + len(line)] = "#" + " " * (len(line) - 1)
            directive = line.rstrip().endswith("\\")
        else:
            directive = False
        start += len(line) + 1
    return "".join(code), "".join(mask)


def normalize(text):
    """Formatting-insensitive form: whitespace runs collapse, and whitespace
    next to punctuation disappears, so `a+b` and `a + b` compare equal."""
    text = re.sub(r"\s+", " ", text).strip()
    return re.sub(r"\s*([^\w\s])\s*", r"\1", text)


def _match(mask, i, open_ch, close_ch):
    """Index of the bracket closing the one at mask[i], or -1."""
    depth = 0
    for j in range(i, len(mask)):
        if mask[j] == open_ch:
            depth += 1
        elif mask[j] == close_ch:
            depth -= 1
            if depth == 0:
                return j
    return -1


def _decl_start(mask, i):
    """Start of the declaration holding offset i: just after the previous
    `;`, `{` or `}`, or after a preprocessor line."""
    j = i
    while j > 0:
        ch = mask[j - 1]
        if ch in ";{}":
            break
        if ch == "\n":
            line_start = mask.rfind("\n", 0, j - 1) + 1
            if mask[line_start:j - 1].lstrip().startswith("#"):
                break
        j -= 1
    return j


_ATTRIBUTE_OPEN = re.compile(r"__attribute__\s*\(\(")
_AFTER_PARAMS_TOKEN = re.compile(
    r"(?:const|noexcept|override|final|volatile)\b|&+|->[\w:<>,\s*&]+")


def _body_open(mask, i):
    """Index of the `{` opening a function body after the parameter list that
    closes just before mask[i], or -1. Only qualifiers, `&`/`&&`,
    `__attribute__((...))` and a trailing return type may come between. A
    scanner, not one regex: a repeated group of overlapping alternatives
    backtracks exponentially on a long run of them."""
    n = len(mask)
    while True:
        while i < n and mask[i].isspace():
            i += 1
        if i == n:
            return -1
        if mask[i] == "{":
            return i
        attr = _ATTRIBUTE_OPEN.match(mask, i)
        if attr:
            close = _match(mask, attr.end() - 2, "(", ")")
            if close < 0:
                return -1
            i = close + 1
            continue
        token = _AFTER_PARAMS_TOKEN.match(mask, i)
        if not token:
            return -1
        i = token.end()


def function_definitions(src, name):
    """[(start, end)] spans of every definition of `name` (end is the closing
    brace, inclusive). Calls and prototypes are not definitions."""
    _, mask = strip_comments(src)
    spans = []
    for m in re.finditer(rf"\b{re.escape(name)}\s*\(", mask):
        prev = mask[max(0, m.start() - 64):m.start()].rstrip()
        if prev.endswith((".", "->", "::")) or re.search(r"\breturn$", prev):
            continue
        close = _match(mask, m.end() - 1, "(", ")")
        if close < 0:
            continue
        body_open = _body_open(mask, close + 1)
        if body_open < 0:
            continue
        body_close = _match(mask, body_open, "{", "}")
        if body_close < 0:
            continue
        start = _decl_start(mask, m.start())
        head = _ATTRS.sub(" ", mask[start:m.start()])
        # `x = f(...) {` or `while (f(...)) {` are not definitions.
        if "=" in head or "(" in head:
            continue
        spans.append((start, body_close))
    return spans


_ATTRS = re.compile(r"template\s*<[^;{}]*?>(?=\s*[\w_])|__attribute__\s*\(\(.*?\)\)|"
                    r"__launch_bounds__\s*\([^)]*\)", re.DOTALL)


def function_hash(src, name):
    code, _ = strip_comments(src)
    spans = function_definitions(src, name)
    if not spans:
        return None
    h = hashlib.sha256()
    for s, e in spans:
        h.update(normalize(code[s:e + 1]).encode())
        h.update(b"\0")
    return h.hexdigest()


def extern_c_definitions(src):
    """{name: normalized signature} for each `extern "C"` function definition
    (prototypes are skipped)."""
    code, mask = strip_comments(src)
    out = {}
    for m in re.finditer(r'\bextern\s*"C"', code):
        stop = re.compile(r"[(;{=]").search(mask, m.end())
        if not stop or stop.group(0) != "(":
            continue  # a data declaration, or an `extern "C" { ... }` block
        paren = stop.start()
        names = _IDENT.findall(mask[m.end():paren])
        if not names:
            continue
        close = _match(mask, paren, "(", ")")
        if close < 0 or _body_open(mask, close + 1) < 0:
            continue
        out[names[-1]] = normalize(code[m.start():close + 1])
    return out


def extern_c_blocks(src):
    """Normalized text of each `extern "C" { ... }` block: anything declared or
    defined inside one has C linkage too."""
    code, mask = strip_comments(src)
    blocks = []
    for m in re.finditer(r'\bextern\s*"C"\s*\{', code):
        close = _match(mask, m.end() - 1, "{", "}")
        blocks.append(normalize(code[m.start():close + 1 if close >= 0 else len(code)]))
    return blocks


# Braces that do not open a scope of their own: a constant inside them is
# still at namespace scope.
_TRANSPARENT_OPEN = re.compile(r'\b(?:namespace\b[\w:\s]*|extern\s*"C"\s*)\{')
_CONST_DECL = (r"(?:\b(?:static|inline)\s+)*\b(?:constexpr|const)\b[^;{{}}()]*?\b"
               r"({name})\s*(?:=(?!=)|\{{)[^;]*;")


@functools.lru_cache(maxsize=4)
def _scope_depth(src):
    """Brace depth at every offset of `src`, not counting namespace and
    `extern "C"` braces (0 = namespace scope)."""
    code, mask = strip_comments(src)
    opens = {m.end() - 1 for m in _TRANSPARENT_OPEN.finditer(code)}
    closes = {_match(mask, i, "{", "}") for i in opens}
    depth, out = 0, []
    for i, ch in enumerate(mask):
        out.append(depth)
        if ch == "{" and i not in opens:
            depth += 1
        elif ch == "}" and i not in closes:
            depth -= 1
    return out


@functools.lru_cache(maxsize=4)
def namespace_constants(src):
    """{name: normalized declaration} of the scalar constexpr/const variables
    declared at namespace scope (the first declaration of each name)."""
    code, mask = strip_comments(src)
    depth = _scope_depth(src)
    decl = re.compile(_CONST_DECL.format(name=r"[A-Za-z_]\w*"))
    out = {}
    pos = 0
    while (m := decl.search(code, pos)) is not None:
        # A `const` parameter with a default value is at depth 0 too, inside
        # the parentheses of a function head.
        head = mask[_decl_start(mask, m.start()):m.start()]
        if depth[m.start()] == 0 and head.count("(") <= head.count(")"):
            out.setdefault(m.group(1), normalize(m.group(0)))
            pos = m.end()
        else:
            pos = m.start() + 1  # the rejected match may have run over a real one
    return out


def constant_declaration(src, name):
    return namespace_constants(src).get(name)


def _reads(src, name):
    """Identifiers a definition of `name` uses outside its literals."""
    _, mask = strip_comments(src)
    ids = set()
    for s, e in function_definitions(src, name):
        ids.update(_IDENT.findall(mask[s:e + 1]))
    return ids


def quoted_includes(src):
    code, _ = strip_comments(src)
    return sorted(set(re.findall(r'^\s*#\s*include\s*"([^"]+)"', code, re.MULTILINE)))


def _macro_targets(src):
    code, _ = strip_comments(src)
    return set(re.findall(r"^\s*#\s*(?:define|undef)\s+([A-Za-z_]\w*)", code, re.MULTILINE))


def manifest(src, functions=(), constants=(), required_includes=("hip_custom_kernels.h",)):
    """Snapshot the frozen contract of the pristine kernel source. Besides the
    listed constants, every namespace-scope constant a frozen function reads is
    frozen. Raises ValueError when a listed gate or constant is not in the
    source."""
    missing_inc = [h for h in required_includes if h not in quoted_includes(src)]
    if missing_inc:
        raise ValueError(f"pristine kernel does not include {missing_inc}")
    fn = {}
    for name in functions:
        h = function_hash(src, name)
        if h is None:
            raise ValueError(f"frozen function '{name}' has no definition in the kernel")
        fn[name] = h
    consts = {}
    for name in constants:
        d = constant_declaration(src, name)
        if d is None:
            raise ValueError(f"frozen constant '{name}' is not declared in the kernel")
        consts[name] = d
    namespace = namespace_constants(src)
    for name in functions:
        for ident in sorted(_reads(src, name) & set(namespace)):
            consts.setdefault(ident, namespace[ident])
    return {
        "extern_c": extern_c_definitions(src),
        "extern_c_blocks": extern_c_blocks(src),
        "functions": fn,
        "constants": consts,
        "required_includes": list(required_includes),
        "exports": None,
    }


def check_source(src, man):
    """List of violations of `man` by the candidate source (empty = OK)."""
    errs = []
    for h in man.get("required_includes", []):
        if h not in quoted_includes(src):
            errs.append(f'#include "{h}" was removed; the frozen ABI header must stay included')
    got = extern_c_definitions(src)
    want = man.get("extern_c", {})
    for name in sorted(set(want) - set(got)):
        errs.append(f'extern "C" {name} was removed or renamed')
    for name in sorted(set(got) - set(want)):
        errs.append(f'extern "C" {name} is new; the exported ABI is frozen')
    for name in sorted(set(want) & set(got)):
        if got[name] != want[name]:
            errs.append(f'extern "C" {name} changed its signature')
    if extern_c_blocks(src) != man.get("extern_c_blocks", []):
        errs.append('extern "C" { ... } blocks changed; the exported ABI is frozen')
    for name, h in sorted(man.get("functions", {}).items()):
        got_h = function_hash(src, name)
        if got_h is None:
            errs.append(f"dispatch gate {name}() was removed")
        elif got_h != h:
            errs.append(f"dispatch gate {name}() changed; only kernel bodies are editable")
    for name, decl in sorted(man.get("constants", {}).items()):
        if constant_declaration(src, name) != decl:
            errs.append(f"frozen constant {name} changed")
    frozen_names = set(want) | set(man.get("functions", {})) | set(man.get("constants", {}))
    for name in sorted(_macro_targets(src) & frozen_names):
        errs.append(f"#define/#undef of frozen name {name}")
    return errs


def exported_c_symbols(lib, nm="nm"):
    """Sorted unmangled function/data symbols `lib` defines in its dynamic
    table. Mangled C++ names are left out: they change with any helper edit and
    are not part of the C ABI."""
    r = subprocess.run([nm, "-D", "--defined-only", "--format=posix", lib],
                       capture_output=True, text=True, check=False)
    if r.returncode != 0:
        raise RuntimeError(f"nm failed on {lib}: {r.stderr.strip()}")
    names = set()
    for line in r.stdout.splitlines():
        parts = line.split()
        if len(parts) >= 2 and parts[1] in _GLOBAL_TYPES and not parts[0].startswith("_"):
            names.add(parts[0].split("@")[0])
    return sorted(names)


_GLOBAL_TYPES = {"T", "W", "V", "D", "B", "R", "i", "u"}


def check_exports(got, want):
    errs = []
    for s in sorted(set(want) - set(got)):
        errs.append(f"kernel library no longer exports {s}")
    for s in sorted(set(got) - set(want)):
        errs.append(f"kernel library exports new symbol {s}")
    return errs
