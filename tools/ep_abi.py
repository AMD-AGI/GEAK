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
  variables       the other namespace-scope variables a frozen function reads
                  (tables, mutable state), by their normalized declarations.
  mutable         those of them that only frozen functions touch; no other
                  code may name them.
  guarded/macros  every identifier a frozen definition uses, and the pristine
                  `#define`/`#undef` lines for them: no macro may be added,
                  changed or dropped on those names.
  env_writes      how many calls set an environment variable; frozen functions
                  read the environment, so there may be no more.
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
        elif c == "'" and _digit_separator(src, i):
            i += 1
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


def _digit_separator(src, i):
    """Whether the `'` at src[i] separates digits (`1'000`, `0x1'F`): it sits
    inside a token that starts with a digit. `u8'a'` and `L'a'` are literals."""
    if i + 1 >= len(src) or not src[i + 1].isalnum():
        return False
    j = i
    while j > 0 and (src[j - 1].isalnum() or src[j - 1] in "_'"):
        j -= 1
    return j < i and src[j].isdigit()


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


# A declarator: a name, any array bounds, then an initializer or the end of the
# declaration. Each repetition starts at a `[` its body cannot hold, so the
# match stays linear.
_VAR_DECLARATOR = re.compile(r"\b([A-Za-z_]\w*)\s*(?:\[[^\[\];{}]*\]\s*)*(=(?!=)|\{|;)")
_NOT_A_VARIABLE = re.compile(r"\s*(?:typedef|using|struct|class|union|enum|namespace|template|"
                             r"friend|static_assert)\b")
_KEYWORDS = {"const", "constexpr", "volatile", "noexcept", "override", "final", "mutable",
             "public", "private", "protected", "default", "delete", "else", "do", "try",
             "return", "break", "continue", "int", "bool", "char", "float", "double", "void"}


@functools.lru_cache(maxsize=4)
def _namespace_declarations(src):
    """{name: [(start, end, mutable)]} of the variables declared at namespace
    scope: scalars, arrays and objects, const or not. A head with parentheses
    is a function head (or a parameter in one) and is skipped; so are heads
    like `__attribute__((aligned(16))) int x;`."""
    _, mask = strip_comments(src)
    depth = _scope_depth(src)
    out = {}
    pos = 0
    while (m := _VAR_DECLARATOR.search(mask, pos)) is not None:
        pos = m.end()
        name = m.group(1)
        if depth[m.start()] != 0 or name in _KEYWORDS:
            continue
        start = _decl_start(mask, m.start())
        head = mask[start:m.start()]
        if (not head.strip() or "(" in head or ")" in head or head.rstrip().endswith(("::", ".", "->"))
                or head.count("<") > head.count(">") or _NOT_A_VARIABLE.match(head)):
            continue
        end = m.end() - 1
        if m.group(2) != ";":
            end = m.end()
            while end < len(mask) and not (mask[end] == ";" and depth[end] == 0):
                end += 1
            end = min(end, len(mask) - 1)
        mutable = not re.search(r"\b(?:const|constexpr)\b", head)
        out.setdefault(name, []).append((start, end, mutable))
        pos = end + 1
    return out


def namespace_variables(src):
    """{name: (normalized declarations, mutable)} of every variable declared at
    namespace scope. All declarations of a name are kept (an `extern` one and
    the definition), and it is mutable when any of them is."""
    code, _ = strip_comments(src)
    return {name: (" ".join(normalize(code[s:e + 1]) for s, e, _ in spans),
                   any(mut for _, _, mut in spans))
            for name, spans in _namespace_declarations(src).items()}


def _reads(src, name):
    """Identifiers a definition of `name` uses outside its literals."""
    _, mask = strip_comments(src)
    ids = set()
    for s, e in function_definitions(src, name):
        ids.update(_IDENT.findall(mask[s:e + 1]))
    return ids


def _names_in(src, name):
    """Every identifier in a definition of `name`, preprocessor lines included:
    a macro that reaches an `#ifdef` inside a frozen body changes it as surely
    as one that reaches a call."""
    code, _ = strip_comments(src)
    ids = set()
    for s, e in function_definitions(src, name):
        ids.update(_IDENT.findall(code[s:e + 1]))
    return ids


def quoted_includes(src):
    code, _ = strip_comments(src)
    return sorted(set(re.findall(r'^\s*#\s*include\s*"([^"]+)"', code, re.MULTILINE)))


def macro_directives(src):
    """{name: [normalized directive]} for every `#define`/`#undef`, in order,
    continuation lines included."""
    code, _ = strip_comments(src)
    out = {}
    lines = code.split("\n")
    i = 0
    while i < len(lines):
        text = lines[i]
        while text.rstrip().endswith("\\") and i + 1 < len(lines):
            i += 1
            text = text.rstrip()[:-1] + " " + lines[i]
        m = re.match(r"\s*#\s*(?:define|undef)\s+([A-Za-z_]\w*)", text)
        if m:
            out.setdefault(m.group(1), []).append(normalize(text))
        i += 1
    return out


_ENV_WRITE = re.compile(r"\b(?:setenv|putenv|unsetenv|_putenv_s|_putenv|_wputenv_s|_wputenv|"
                        r"SetEnvironmentVariable[AW]?)\s*\(")


def env_writes(src):
    """How many calls in `src` set or clear an environment variable."""
    _, mask = strip_comments(src)
    return len(_ENV_WRITE.findall(mask))


def _uses_outside(src, name, spans):
    """Whether `name` appears in `src` (macro bodies included) outside `spans`
    and outside its own namespace-scope declarations."""
    code, _ = strip_comments(src)
    allowed = list(spans) + [(s, e) for s, e, _ in _namespace_declarations(src).get(name, [])]
    return any(not any(s <= m.start() <= e for s, e in allowed)
               for m in re.finditer(rf"\b{re.escape(name)}\b", code))


def manifest(src, functions=(), constants=(), required_includes=("hip_custom_kernels.h",)):
    """Snapshot the frozen contract of the pristine kernel source. Besides the
    listed constants, every namespace-scope variable a frozen function reads is
    frozen: scalar constants under `constants`, the rest (tables, mutable
    state) under `variables`. A mutable one that only frozen functions touch is
    also listed in `mutable` and must stay theirs. `guarded` is every
    identifier a frozen definition uses; `macros` holds the pristine
    `#define`/`#undef` lines for those names. Raises ValueError when a listed
    gate or constant is not in the source."""
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
    reads = set().union(*(_reads(src, name) for name in functions))
    for ident in sorted(reads & set(namespace)):
        consts.setdefault(ident, namespace[ident])
    declared = namespace_variables(src)
    variables = {n: declared[n][0] for n in sorted(reads & set(declared)) if n not in consts}
    spans = [sp for name in functions for sp in function_definitions(src, name)]
    # A mutable variable an editable function also touches cannot be held to
    # the frozen functions; its declaration is still frozen.
    mutable = [n for n in variables if declared[n][1] and not _uses_outside(src, n, spans)]
    extern_c = extern_c_definitions(src)
    guarded = set(extern_c) | set(fn) | set(consts) | set(variables)
    guarded = guarded.union(*(_names_in(src, name) for name in functions))
    macros = macro_directives(src)
    return {
        "extern_c": extern_c,
        "extern_c_blocks": extern_c_blocks(src),
        "functions": fn,
        "constants": consts,
        "variables": variables,
        "mutable": mutable,
        "guarded": sorted(guarded),
        "macros": {n: macros[n] for n in sorted(guarded & set(macros))},
        "env_writes": env_writes(src),
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
    declared = namespace_variables(src) if man.get("variables") else {}
    for name, decl in sorted(man.get("variables", {}).items()):
        if declared.get(name, (None,))[0] != decl:
            errs.append(f"frozen variable {name} changed")
    if man.get("mutable"):
        spans = [sp for name in man.get("functions", {}) for sp in function_definitions(src, name)]
        for name in man["mutable"]:
            if _uses_outside(src, name, spans):
                errs.append(f"frozen variable {name} is used outside the frozen functions; "
                            "only they may read or write it")
    guarded = (set(man.get("guarded", ())) | set(want) | set(man.get("functions", {}))
               | set(man.get("constants", {})))
    want_macros = man.get("macros", {})
    got_macros = macro_directives(src)
    for name in sorted(guarded & (set(got_macros) | set(want_macros))):
        if got_macros.get(name, []) != want_macros.get(name, []):
            errs.append(f"#define/#undef of {name} changed; a frozen function uses that name")
    if env_writes(src) > man.get("env_writes", 0):
        errs.append("the kernel sets an environment variable; frozen functions read the "
                    "environment, so it may not change")
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
