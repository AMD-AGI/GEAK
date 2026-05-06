#!/usr/bin/env python3
"""CI gate: no literal `kernel_type == "triton"|"hip"` outside `kernel_languages/`.

Part of PR-1 (Foundation + Cleanup) per docs/refactor/EXECUTION_PLAN.md §9.1.

Semantic-site audit (see INVARIANTS.md + CODEBASE_AUDIT.md §9 for the 16 sites):
This script catches the narrow pattern (literal equality comparisons). The broader
semantic sites (dict dispatch, prompt templates, _LANGUAGE_GUIDANCE maps) are
tracked separately by inspection during PR-2 and PR-3.

FAIL-strict: literal-equality language checks are forbidden outside
``kernel_languages/``.  Core code MUST route through
``kernel_languages.registry`` for language detection.  If you land a
new violation, either move the check into a new/existing language
bundle or use ``registry.detect_best()`` / ``registry.detect_best_by_name()``.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = REPO_ROOT / "src" / "minisweagent"
KERNEL_LANGUAGES_DIR = SRC_DIR / "kernel_languages"

# The literal-equality patterns we forbid in core code.
LEAK_PATTERNS = [
    re.compile(r'kernel_type\s*==\s*["\'](?:triton|hip)["\']'),
    re.compile(r'kernel_language\s*==\s*["\'](?:triton|hip)["\']'),
    re.compile(r'["\'](?:triton|hip)["\']\s*==\s*kernel_type'),
    re.compile(r'["\'](?:triton|hip)["\']\s*==\s*kernel_language'),
]


def is_in_kernel_languages(path: Path) -> bool:
    try:
        path.relative_to(KERNEL_LANGUAGES_DIR)
        return True
    except ValueError:
        return False


def main() -> int:
    violations: list[tuple[Path, int, str]] = []

    for py in SRC_DIR.rglob("*.py"):
        if is_in_kernel_languages(py):
            continue
        try:
            text = py.read_text()
        except (UnicodeDecodeError, OSError):
            continue
        for i, line in enumerate(text.splitlines(), start=1):
            for pat in LEAK_PATTERNS:
                if pat.search(line):
                    violations.append((py.relative_to(REPO_ROOT), i, line.strip()))

    if not violations:
        print("[OK] No literal kernel_type==\"triton\"|\"hip\" leaks outside kernel_languages/")
        return 0

    # FAIL-strict: any literal-equality language check outside
    # kernel_languages/ breaks the build.  The intended mitigation for
    # any new violation is always:
    #   1. Move the check into a language bundle (per-language data), OR
    #   2. Use ``kernel_languages.registry.detect_best()`` /
    #      ``detect_best_by_name()`` for the routing.
    print(f"[FAIL] {len(violations)} literal language-equality leak(s) "
          f"in core code:")
    for p, lineno, line in violations:
        print(f"  {p}:{lineno}  {line}")
    print(
        "\nHint: route language detection through "
        "``kernel_languages.registry`` instead of literal equality "
        "against a language name."
    )
    return 1


if __name__ == "__main__":
    sys.exit(main())
