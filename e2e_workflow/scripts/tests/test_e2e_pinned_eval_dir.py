#!/usr/bin/env python3
"""test_e2e_pinned_eval_dir.py — a pinned eval dir is the one the run uses.

run_e2e pins EVAL_DIR, writes into it before the workflow starts, and reads workflow_return.json and
recovers results from disk only there. The Director used to apply the "exists & non-empty -> append
_${RANDOM}" rule to the pinned dir too, built a sibling, and the workflow took the Director's answer as
EVAL_DIR. Every later stage then wrote where run_e2e never looked, and a multi-hour run surfaced as
workflow_parse_error. So the Director must keep the pin as-is, and the workflow must refuse any other.

    python3 e2e_workflow/scripts/tests/test_e2e_pinned_eval_dir.py
"""
import os
import re
import sys
from pathlib import Path

HERE = os.path.dirname(os.path.abspath(__file__))
E2E = os.path.abspath(os.path.join(HERE, "..", ".."))
SRC = Path(E2E, "e2e_workflow.js").read_text()
DIRECTOR = Path(E2E, "roles", "director.md").read_text()

FAILED = []


def check(name, cond, detail=""):
    print(f"  {'ok  ' if cond else 'FAIL'}  {name}" + (f"   {detail}" if detail and not cond else ""))
    if not cond:
        FAILED.append(name)


setup_start = SRC.index("roleAgent('director', 'setup'")
setup_assign = SRC.index("EVAL_DIR = setup.eval_dir;", setup_start)
setup_block = SRC[setup_start:setup_assign]

check(
    "the workflow refuses a Director eval_dir that differs from the pin",
    re.search(r"if \(EVAL_DIR_OVERRIDE && trimSlash\(setup\.eval_dir\) !== trimSlash\(EVAL_DIR_OVERRIDE\)\)", setup_block)
    is not None
    and "throw new Error(`Setup failed: Director built eval_dir" in setup_block,
    "expected a pin check that throws before EVAL_DIR = setup.eval_dir",
)

check(
    "trailing slashes do not count as a different dir",
    "const trimSlash = (p) => String(p || '').trim().replace(/\\/+$/, '');" in setup_block,
)

step2 = DIRECTOR[DIRECTOR.index("2. Decide `EVAL_DIR`") : DIRECTOR.index("3. Build the layout")]
override_rule, _, default_rule = step2.partition("- Otherwise")

check(
    "the Director keeps a pinned dir even when it exists and holds files",
    "exactly that path, even when it already exists" in override_rule and "Never rename or suffix it" in override_rule,
)

check(
    "the append-_${RANDOM} rule applies only to the generated default",
    "_${RANDOM}" not in override_rule and "_${RANDOM}" in default_rule,
)

if FAILED:
    print(f"\n{len(FAILED)} check(s) failed")
    sys.exit(1)
print("\nall checks passed")
