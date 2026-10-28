"""Root pytest configuration.

A bare `pytest` at the repo root used to abort before running anything. Two unrelated
causes, kept apart below because they want opposite follow-up.

1. SCRIPT_STYLE -- contract checks meant to be run as programs: they assert at import and
   end in `sys.exit(...)`, which fires during collection as `INTERNALERROR> SystemExit: 0`
   and kills the entire run. Still fully checked -- `ci/test_standalone_contracts.py` runs
   each as a subprocess and fails if this list grows past what it runs. Ignoring them
   without that would trade a loud failure for a quiet hole.

2. EXAMPLE_TASKS -- workload payload for the workflow to operate on, not this repo's
   tests. `knn/test_knn.py` opens a source file by relative path at import, and needs a
   HIP toolchain and a GPU besides. Deliberately run nowhere: a GPU test belongs behind
   `gpu_lock.sh`.

Both lists name files, never directories: `kernel_workflow/scripts/` also holds six
ordinary test files CI does collect, and a directory-wide ignore would silently drop them.
"""

# Run as programs by test_standalone_contracts.py. Keep in sync with CONTRACT_CHECKS there.
SCRIPT_STYLE = [
    "kernel_workflow/scripts/test_kb.py",
    "kernel_workflow/scripts/test_kb_switch.py",
    "kernel_workflow/scripts/test_lane_refs.py",
    "e2e_workflow/scripts/tests/test_e2e_lane_defaults.py",
]

# Workload payload, not repo tests. Needs a HIP build and a GPU; deliberately unrun here.
EXAMPLE_TASKS = [
    "examples/tasks/knn/test_knn.py",
]

collect_ignore = [*SCRIPT_STYLE, *EXAMPLE_TASKS]
