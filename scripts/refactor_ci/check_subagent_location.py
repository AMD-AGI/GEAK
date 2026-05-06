#!/usr/bin/env python3
"""CI gate: any class subclassing `SubagentBase` must live under `subagents/`.

Part of PR-1 (Foundation + Cleanup) per docs/refactor/EXECUTION_PLAN.md §7 Principle #9.

FAIL-strict from the moment SubagentBase is introduced. Before then, there are
zero subclasses so this script is a no-op — becomes active as soon as PR-1's
subagents/base.py lands.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = REPO_ROOT / "src" / "minisweagent"
SUBAGENTS_DIR = SRC_DIR / "subagents"

# Class definition pattern that inherits from SubagentBase (or imports-then-uses it)
SUBCLASS_RE = re.compile(r"^\s*class\s+\w+\s*\(\s*SubagentBase\s*[,)]")


def is_under_subagents(path: Path) -> bool:
    try:
        path.relative_to(SUBAGENTS_DIR)
        return True
    except ValueError:
        return False


def main() -> int:
    violations: list[tuple[Path, int, str]] = []

    for py in SRC_DIR.rglob("*.py"):
        try:
            text = py.read_text()
        except (UnicodeDecodeError, OSError):
            continue
        if "SubagentBase" not in text:
            continue
        # Ignore imports/type hints — only flag actual class definitions
        for i, line in enumerate(text.splitlines(), start=1):
            if SUBCLASS_RE.search(line) and not is_under_subagents(py):
                violations.append((py.relative_to(REPO_ROOT), i, line.strip()))

    if not violations:
        print("[OK] All SubagentBase subclasses live under src/minisweagent/subagents/")
        return 0

    print(f"[FAIL] {len(violations)} SubagentBase subclass(es) outside subagents/:")
    for p, lineno, line in violations:
        print(f"  {p}:{lineno}  {line}")
    return 1


if __name__ == "__main__":
    sys.exit(main())
