"""Root pytest configuration.

A bare `pytest` at the repo root used to abort before running anything. Two unrelated
causes, kept in two lists below because they want opposite follow-up.

1. SCRIPT_STYLE -- four test files run their assertions at import time and finish with a
   module-level `sys.exit(...)`. That is a deliberate shape; they are contract checks
   meant to be run as programs. But pytest cannot express it: the `sys.exit(0)` fires
   during collection and pytest reports

       INTERNALERROR> SystemExit: 0

   which kills the ENTIRE run, including the thousands of ordinary tests already
   collected. These stay fully checked -- `ci/test_standalone_contracts.py` runs each as a
   subprocess and asserts a zero exit, the same way CI's "KB and workflow contract
   checks" step does, and a test there fails if this list ever grows past what it runs.
   Ignoring them without that would have traded a loud failure for a quiet hole.

2. EXAMPLE_TASKS -- `examples/tasks/` is workload payload for the workflow to operate
   on, not part of this repo's test suite. `knn/test_knn.py` opens `src/knn_cuda.hip`
   by RELATIVE path at import, so it raises FileNotFoundError during collection from any
   other directory, and even from its own directory it needs a HIP toolchain and a GPU.
   Unlike group 1 these are NOT run elsewhere, and should not be: a GPU test belongs
   behind `gpu_lock.sh`, not in the default suite.

Scope both lists precisely. `kernel_workflow/scripts/` also holds six ordinary
pytest-collectable files that CI DOES collect (test_detect_language, test_upstream_knobs,
test_capture_validation_env, test_check_flydsl_compat, and two under `scripts/tests/`), so
a directory-wide ignore would silently drop them from every local run.

To re-derive group 1 after adding tests:

    python3 -c "
    import ast,pathlib
    for p in sorted(pathlib.Path('.').rglob('test_*.py')):
        t=ast.parse(p.read_text())
        for n in t.body:
            if isinstance(n,ast.If) and getattr(n.test.left,'id','')=='__name__': continue
            if any(isinstance(m,ast.Call) and getattr(m.func,'attr','')=='exit' for m in ast.walk(n)):
                print(p); break
    "
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
