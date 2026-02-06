#!/usr/bin/env python3
"""Example runner: delegates to kernel-profile (geakagent.kernel_profile)."""
import sys
from pathlib import Path

# Ensure src is on path so geakagent is importable when run as script
_repo_root = Path(__file__).resolve().parent.parent.parent
_src = _repo_root / "src"
if str(_src) not in sys.path:
    sys.path.insert(0, str(_src))

from geakagent.kernel_profile import main

if __name__ == "__main__":
    main()
