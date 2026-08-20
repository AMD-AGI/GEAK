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


def used_in_code(path):
    """ALL-CAPS identifiers in CODE position: comments, strings and object keys stripped first."""
    s = open(path).read()
    s = re.sub(r'//[^\n]*', '', s)
    s = re.sub(r'/\*.*?\*/', '', s, flags=re.S)
    s = re.sub(r'`(?:[^`\\]|\\.)*`', '``', s)
    s = re.sub(r"'(?:[^'\\]|\\.)*'", "''", s)
    s = re.sub(r'"(?:[^"\\]|\\.)*"', '""', s)
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
