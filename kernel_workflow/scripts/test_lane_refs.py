#!/usr/bin/env python3
"""test_lane_refs.py — every ALL-CAPS constant `kernel_lane.js` uses is declared IN `kernel_lane.js`.

A parse test cannot catch this class of bug. `kernel_lane.js` compiles fine with a reference to a
constant that only exists in `kernel_workflow.js`; the ReferenceError arrives when the line RUNS,
which for a plan-time constant is several minutes into a kernel. It happened: a KB budget block
ported from the dispatcher referenced `USE_LEARNED_READ`, and 13 kernels died one plan step in
before the run was stopped. The two files look alike and code moves between them, so the specific
hazard is a constant that resolves in the file it was copied FROM.

    python3 kernel_workflow/scripts/test_lane_refs.py
"""
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
WF = os.path.dirname(HERE)


def declared(path):
    return set(re.findall(r'^\s*(?:const|let|var)\s+([A-Z][A-Z0-9_]{2,})\s*=',
                          open(path).read(), re.M))


_REGEX_OK_AFTER = set("(,=:[!&|?{};+-*%~^<>") | {""}


def code_only(src):
    """Blank comments, strings, template literals and regex literals in ONE left-to-right pass.

    A pass-per-construct cascade cannot do this. Each pass re-scans the whole file, so any
    construct holding a delimiter belonging to a LATER pass — a quote inside a regex character
    class, an apostrophe in template text — shifts every boundary after it, and the remainder of
    the file is read against the wrong state. That is not hypothetical here: the cascade this
    replaced reported 14 undeclared constants, all of them prose and shell words (`NOT`, `THIS`,
    `SSL_CERT_FILE`) lifted out of the KB_ENV_PRELUDE strings it had lost track of. A check that
    cries wolf 14 times is a check nobody reads the 15th time, when the wolf is real.
    """
    out, i, n, prev = [], 0, len(src), ""
    # Frames: "tmpl" = inside template TEXT; an int = inside a ${...} substitution, counting the
    # brace depth of object literals so their `}` is not mistaken for the end of the substitution.
    stack = []
    while i < n:
        ch, two = src[i], src[i:i + 2]

        if stack and stack[-1] == "tmpl":
            if ch == "\\":
                i += 2
            elif ch == "`":
                stack.pop()
                i += 1
                prev = '"'
            elif two == "${":
                stack.append(0)
                i += 2
                prev = "("
            else:
                i += 1
            out.append(" ")
            continue

        if two == "//":
            while i < n and src[i] != "\n":
                i += 1
            out.append(" ")
            continue
        if two == "/*":
            i += 2
            while i < n and src[i:i + 2] != "*/":
                i += 1
            i += 2
            out.append(" ")
            continue
        if ch in "\"'":
            quote, i = ch, i + 1
            while i < n and src[i] != quote:
                i += 2 if src[i] == "\\" else 1
            i += 1
            out.append(' "" ')
            prev = '"'
            continue
        if ch == "`":
            stack.append("tmpl")
            i += 1
            out.append(' "" ')
            prev = '"'
            continue
        if stack and isinstance(stack[-1], int):
            if ch == "{":
                stack[-1] += 1
            elif ch == "}":
                if stack[-1] == 0:
                    stack.pop()
                    i += 1
                    out.append(" ")
                    prev = '"'
                    continue
                stack[-1] -= 1
        if ch == "/" and prev in _REGEX_OK_AFTER:
            i += 1
            in_class = False
            while i < n:
                c = src[i]
                if c == "\\":
                    i += 2
                    continue
                if c == "[":
                    in_class = True
                elif c == "]":
                    in_class = False
                elif (c == "/" and not in_class) or c == "\n":
                    break
                i += 1
            i += 1
            while i < n and src[i] in "gimsuyd":
                i += 1
            out.append(' "" ')
            prev = '"'
            continue
        out.append(ch)
        if not ch.isspace():
            prev = ch
        i += 1
    return "".join(out)


def used_in_code(path):
    """ALL-CAPS identifiers in CODE position: comments, strings and object keys removed first."""
    s = code_only(open(path).read())
    keys = set(re.findall(r'\b([A-Z][A-Z0-9_]{2,})\s*:', s))   # {KEY: value} is a key, not a read
    return set(re.findall(r'\b([A-Z][A-Z0-9_]{2,})\b', s)) - keys


BUILTINS = {'JSON', 'Math', 'Object', 'Array', 'String', 'Number', 'Boolean', 'Promise',
            'Set', 'Map', 'Error', 'NaN', 'Infinity', 'RegExp', 'Date', 'Symbol', 'BigInt'}

lane = os.path.join(WF, 'kernel_lane.js')
disp = os.path.join(WF, 'kernel_workflow.js')
lane_decl, disp_decl = declared(lane), declared(disp)
missing = sorted(used_in_code(lane) - lane_decl - BUILTINS)

leaked = [m for m in missing if m in disp_decl]
other = [m for m in missing if m not in disp_decl]

for m in leaked:
    print(f"  FAIL  {m}: used in kernel_lane.js, declared only in kernel_workflow.js "
          f"— it will throw ReferenceError when that line runs")
for m in other:
    print(f"  FAIL  {m}: used in kernel_lane.js and declared nowhere")
if not missing:
    print(f"  ok    every ALL-CAPS constant kernel_lane.js reads is declared there "
          f"({len(lane_decl)} declarations checked)")
    sys.exit(0)
sys.exit(1)
