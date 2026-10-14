#!/usr/bin/env python3
"""Run a frozen task's entry script with the standard library intact.

An extracted task directory contains `unittest.py` (its entry) and `cases.py`. Running it
the obvious way -- `cd <task> && python3 unittest.py` -- puts the task directory at
`sys.path[0]`, so the task's own `unittest.py` **shadows the standard library package of
the same name**. Nothing fails at import time; it fails later and from a distance, when
something inside torch reaches for a submodule:

    ModuleNotFoundError: No module named 'unittest.mock'; 'unittest' is not a package

which is what `torch.load` does on this stack. The measurement then returns no numbers at
all, and the attempt is reverted for the wrong reason.

The fix is ordering, not patching: import the shadowed stdlib modules **before** the task
directory is placed on the path, so `sys.modules` already holds the real ones. The entry
is then executed as `__main__` with the task directory as cwd and `sys.argv[0]` set to
it, which is exactly what it would see when run directly.

The alternative -- a `sitecustomize.py` on PYTHONPATH -- has a trap worth recording: a
candidate overlay may ship its own `sitecustomize.py`, and only the first one on the path
is imported, so the shim can silently disable the overlay and measure the baseline twice.

Usage: python3 run_task_entry.py <task_dir> [entry.py] [args...]
"""
import os
import runpy
import sys

# Before anything else touches sys.path: pull the stdlib names a task directory can
# plausibly shadow into sys.modules. A later `import unittest.mock` then resolves from
# the cache instead of from the task directory.
import unittest  # noqa: F401
import unittest.mock  # noqa: F401


def main(argv):
    if not argv:
        print(__doc__.strip().splitlines()[-1], file=sys.stderr)
        return 2
    task_dir = os.path.abspath(argv[0])
    entry = argv[1] if len(argv) > 1 else "unittest.py"
    rest = list(argv[2:])
    path = os.path.join(task_dir, entry)
    if not os.path.exists(path):
        print("no such entry: %s" % path, file=sys.stderr)
        return 2

    os.chdir(task_dir)
    # The task directory still goes on the path -- the entry imports cases.py and
    # harness_lib.py from it -- but only after the stdlib is already resolved.
    sys.path.insert(0, task_dir)
    sys.argv = [path] + rest
    runpy.run_path(path, run_name="__main__")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
