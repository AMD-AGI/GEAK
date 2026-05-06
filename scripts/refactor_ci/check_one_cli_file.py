#!/usr/bin/env python3
"""CI gate: only ``src/minisweagent/cli.py`` may define a Typer app.

Part of PR-1 (Foundation + Cleanup) per docs/refactor/EXECUTION_PLAN.md
§7 Principle #8.

This gate is now FAIL-strict: if anyone reintroduces a Typer app (``typer.Typer``
constructor or ``@app.command`` decorator) outside the canonical ``cli.py``,
the build fails.  All legacy auxiliary CLIs that once had their own Typer app
(``run/mini.py``, ``run/mini_extra.py``, ``run/inspector.py``, ``run/github_issue.py``,
``run/extra/swebench*.py``, ``run/extra/config.py``, ``tools/strategy_manager.py``)
were either consolidated into ``cli.py`` or had their dead Typer wrappers
deleted.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = REPO_ROOT / "src" / "minisweagent"
CANONICAL_CLI = SRC_DIR / "cli.py"

# Patterns that declare a Typer app at module level
TYPER_DECL = re.compile(r"^\s*(?:\w+\s*=\s*)?typer\.Typer\s*\(", re.MULTILINE)
APP_COMMAND = re.compile(r"^\s*@app\.command\s*\(", re.MULTILINE)

# Files we allow to have Typer decls (the canonical one)
ALLOWED = {CANONICAL_CLI.relative_to(REPO_ROOT)}


def main() -> int:
    violators: list[tuple[Path, str]] = []

    for py in SRC_DIR.rglob("*.py"):
        rel = py.relative_to(REPO_ROOT)
        if rel in ALLOWED:
            continue
        try:
            text = py.read_text()
        except (UnicodeDecodeError, OSError):
            continue
        m = TYPER_DECL.search(text) or APP_COMMAND.search(text)
        if m:
            violators.append((rel, m.group(0).strip()))

    if violators:
        print(
            f"[FAIL] {len(violators)} Typer app(s) outside the canonical "
            f"src/minisweagent/cli.py:"
        )
        for p, snippet in violators:
            print(f"  {p} :: {snippet}")
        print(
            "\nHint: move the commands into cli.py as @app.command() subcommands, "
            "or delete the Typer wrapper if the module only needs to export plain "
            "helper functions (see run/extra/config.py for an example)."
        )
        return 1

    print("[OK] No Typer apps outside src/minisweagent/cli.py")
    return 0


if __name__ == "__main__":
    sys.exit(main())
