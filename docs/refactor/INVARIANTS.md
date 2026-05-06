# GEAK Pipeline Invariants (log markers — smoke-test enforced)

These markers were captured from two live runs on `origin/main @ d7b880c3` (2026-04-22). Smoke tests assert their presence on every PR. If a refactor PR removes a marker, the PR must either (a) add a documented replacement marker in the same commit, or (b) fail the smoke test.

Source of truth: `docs/refactor/CODEBASE_AUDIT.md` §0.1 + §14.

## Triton heterogeneous path — 12 invariant markers

From `/data/sapmajum/triton_runs/gemm_a16w16_atomic_canonical-rocm700_memon_20260422_083415.log`:

```
Normalized kernel_type from task content: triton                      [line 34]
--- Step 1/7: Resolve kernel URL                                      [line 76]
--- Step 2/7: Codebase context                                        [line 89]
--- Step 3/7: Test discovery                                          [line 95]
--- Step 4/7: Baseline                                                [line 126]
--- Step 5/7: Kernel profiling                                        [line 155]
--- Step 6/7: Baseline metrics                                        [line 185]
--- Step 7/7: Commandment                                             [line 192]
Using heterogeneous mode based on discovery.                          [line 197]
run_orchestrator:                                                     [line 200]
start_round=1, heterogeneous=True                                     [line 202]  (continuation)
Cross-session memory                                                  [line 207]
Exploration Phase (this may take a few minutes)                       [line 215]
```

**Note on formatting**: the original audit used `--- Step N/7: Title ---` with trailing dashes and CapCase Titles. Live log inspection shows the real format is `--- Step N/7: <title>` with lowercase second word and NO trailing `---`. Step 4 is "Baseline", not "Harness Validation" (the audit was wrong; harness validation is folded into Step 4). The smoke tests use tolerant patterns that match both forms.

Regex list used by `tests/smoke/test_triton_hetero_invariants.py`:

```python
EXPECTED_MARKERS_TRITON = [
    r"Normalized kernel_type from task content: triton",
    r"--- Step 1/7: Resolve kernel URL ---",
    r"--- Step 2/7: Codebase Context ---",
    r"--- Step 3/7: Test Discovery ---",
    r"--- Step 4/7: Harness Validation ---",
    r"--- Step 5/7: Kernel Profiling ---",
    r"--- Step 6/7: Baseline Metrics ---",
    r"--- Step 7/7: Commandment ---",
    r"Using heterogeneous mode based on discovery",
    r"run_orchestrator:.*heterogeneous=True",
    r"Cross-session memory",
    r"Exploration Phase",
]
```

After PR-2 (phase-based preprocess), Step markers become `Phase:` equivalents. PR-2 emits BOTH forms during transition window.

## HIP homogeneous path — 11 invariant markers

From `/data/sapmajum/AgentKernelArena/logs/hip_ab_v3/assign_score_withk_mem_20260415_180639.log`:

```
Normalized kernel_type from task content: hip               [line 13]
--- Step 1/7: Resolve kernel URL ---                        [line 47]
--- Step 5/7: Kernel Profiling ---                          [line 54]   # steps 2-4 legitimately skipped today
--- Step 6/7: Baseline Metrics ---                          [line 70]
--- Step 7/7: Commandment ---                               [line 77]
Using homogeneous mode based on discovery                   [line 82]
Retriever: category=unknown language=hip                    [line 85]
Cross-session memory injected into homogeneous task (27608 chars)  [line 90]
Homogeneous Agent (2 agents, GPUs [2, 3])                   [line 96]
Sub-agent 1 (task_1) started on GPU 3                       [line 106]
Sub-agent 0 (task_0) started on GPU 2                       [line 107]
```

Regex list used by `tests/smoke/test_hip_homo_invariants.py`:

```python
EXPECTED_MARKERS_HIP = [
    r"Normalized kernel_type from task content: hip",
    r"--- Step 1/7: Resolve kernel URL ---",
    r"--- Step 5/7: Kernel Profiling ---",
    r"--- Step 6/7: Baseline Metrics ---",
    r"--- Step 7/7: Commandment ---",
    r"Using homogeneous mode based on discovery",
    r"Retriever: category=\w+ language=hip",
    r"Cross-session memory injected into homogeneous task \(\d+ chars\)",
    r"Homogeneous Agent \(\d+ agents, GPUs \[[^]]+\]\)",
    r"Sub-agent \d+ \(task_\d+\) started on GPU \d+",
    r"Sub-agent \d+ \(task_\d+\) started on GPU \d+",   # >=2 sub-agents in parallel
]
```

## Known asymmetries (fixed by PR-2 + PR-3)

- **HIP silently skips steps 2-4** (preprocess). Bug #1 in EXECUTION_PLAN.md §0.3 — PR-2 rewrites preprocess as phases that run uniformly across languages.
- **HIP does NOT write to memory.db** (no `record_optimization_outcome` call on homo path). Bug #6 — PR-3 makes KB write uniform across modes.
- **HIP does NOT run FULL_BENCHMARK per round**. Bug #1 — PR-3 `run_pipeline` enforces per-round FULL_BENCHMARK uniformly.

## Regression thresholds

See `tests/regression/baseline_speedups.yaml` for the 13-kernel speedup floor per EXECUTION_PLAN.md §0.2.
