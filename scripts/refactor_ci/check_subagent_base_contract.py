#!/usr/bin/env python3
"""CI gate: every SubagentBase subclass overrides EXACTLY ONE of `run` / `loop`.

Part of PR-1 (Foundation + Cleanup) per docs/refactor/EXECUTION_PLAN.md §16.2.

The two execution methods:
  - run(**inputs) -> str | dict             (one-shot)
  - loop(max_attempts, verify_fn, **inputs) (multi-round)

A SubagentBase subclass that overrides neither, or both, is a design error.

Uses AST walk (not regex) because correctness matters and method definitions
can span multiple lines.

FAIL-strict from day one. Pre-PR-1 with zero subclasses: no-op.
"""

from __future__ import annotations

import ast
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
SUBAGENTS_DIR = REPO_ROOT / "src" / "minisweagent" / "subagents"


def finds_subagent_subclass(cls: ast.ClassDef) -> bool:
    """Returns True if cls inherits from SubagentBase (by simple name match)."""
    for base in cls.bases:
        if isinstance(base, ast.Name) and base.id == "SubagentBase":
            return True
        if isinstance(base, ast.Attribute) and base.attr == "SubagentBase":
            return True
    return False


def overridden_methods(cls: ast.ClassDef) -> set[str]:
    """Method names defined directly in cls (not inherited)."""
    return {
        node.name
        for node in cls.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    }


def main() -> int:
    if not SUBAGENTS_DIR.exists():
        print("[OK] No subagents/ directory yet — nothing to check")
        return 0

    violations: list[tuple[Path, str, str]] = []

    for py in SUBAGENTS_DIR.rglob("*.py"):
        if py.name == "base.py":
            continue  # SubagentBase itself is allowed to define both
        try:
            tree = ast.parse(py.read_text())
        except (SyntaxError, UnicodeDecodeError, OSError):
            continue

        for node in ast.walk(tree):
            if not isinstance(node, ast.ClassDef):
                continue
            if not finds_subagent_subclass(node):
                continue
            methods = overridden_methods(node)
            has_run = "run" in methods
            has_loop = "loop" in methods
            if has_run and has_loop:
                violations.append((
                    py.relative_to(REPO_ROOT),
                    node.name,
                    "overrides BOTH run() and loop() — must override exactly ONE"
                ))
            elif not has_run and not has_loop:
                violations.append((
                    py.relative_to(REPO_ROOT),
                    node.name,
                    "overrides NEITHER run() nor loop() — must override exactly ONE"
                ))

    if not violations:
        print("[OK] All SubagentBase subclasses override exactly one of run/loop")
        return 0

    print(f"[FAIL] {len(violations)} SubagentBase subclass(es) violate the contract:")
    for p, name, reason in violations:
        print(f"  {p}  class {name}: {reason}")
    return 1


if __name__ == "__main__":
    sys.exit(main())
