#!/usr/bin/env python3
"""The standard recipe entry point: `python3 workload/run.py --config <scenario> all`.

Everything it can do, and every flag it takes, is `engine/core/stage.py` — see its module
docstring, or `workload/recipes/standard.md`. The adaptive loop is the root SKILL.md.
This file exists so that the recipe entry point does not
move when the implementation behind it does.
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "engine", "core"))
import stage  # noqa: E402

if __name__ == "__main__":
    sys.exit(stage.main())
