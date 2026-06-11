# Implementation plan: team_workflow (kernel-optimization orchestration as a dynamic workflow)

## Goal
Refactor the existing markdown-driven `team` skill into a dynamic workflow orchestrated by a
**deterministic JS Workflow script**, beating the old skill's 36.5x geomean on knn by >50% (≈55x).

## Output directory
`/wekafs/zihao/2026/geak_cc/PerfSkills/workflows/` (user-confirmed; do not modify any existing files —
scripts/knowledge are fresh copies).

## Confirmed architectural enhancements
- **A. Engineers specialized by persona**: algorithm / memory / compute / host_runtime — four personas,
  each loading only its relevant knowledge for a focused context.
- **B. Host/Runtime as a first-class role**: focused on the launch-overhead floor, dispatch
  merging/fusion, CUDA graph, host bypass — the key lever to break 55x on knn, reused for e2e too.
- **C. Cross-round insight blackboard + hypothesis ledger**: each round the TechLead distills "what we
  learned" and injects it into the next round's engineers; each direction is a hypothesis with an
  expected gain, and after verify we record actual vs expected to guide re-planning.
- **E. Integrator** (upgraded merge): may manually rewrite conflicting good ideas into one coherent best
  implementation, not just stack `git apply` patches.
- **H. Director arbitration / send-back**: at final acceptance, if a flag fires (verified ≪ claimed,
  etc.) the Director may send it back to the TechLead for one corrective round.
- Not adopted: D red-team, F beam search, G propose-then-dedup (directions are planned orthogonally by
  the TechLead, so dedup happens at planning time).
- Keep the **independent re-measuring verify_engineer** (core reliability, not D). Greedy single-winner
  commit each round.

## Directory structure
```
workflows/
├── README.md
├── team_workflow.js              # deterministic Workflow script
├── roles/
│   ├── director.md               # setup + final independent validation + arbitration/send-back (H)
│   ├── tech_lead.md              # analysis/roadmap + per-round re-plan (insight blackboard / hypothesis ledger C, diversity check) + integration guidance + final report
│   ├── engineer.md               # optimization worker, specialized by persona (A), incl. host_runtime (B) + self_monitoring
│   ├── benchmark_engineer.md     # harness / COMMANDMENT / baseline
│   ├── profile_engineer.md       # profile + bottleneck classification
│   ├── verify_engineer.md        # independent re-measurement (source of trust)
│   └── integrator.md             # merge/rewrite multiple winning patches (E)
├── knowledge/
│   ├── optimization_strategies.md
│   ├── hip_optimization.md
│   ├── triton_optimization.md
│   ├── wrapper_optimization.md   # primary for the host_runtime persona
│   ├── profiling_guide.md
│   ├── amd_mi300x.md
│   ├── self_monitoring.md
│   └── geomean_levers.md         # new: launch floor / slowest case / dispatch merging / fusion / CUDA graph
└── scripts/
    ├── gpu_lock.sh               # copied over
    └── profile_kernel.sh         # copied over
```

## Role → workflow mapping
- **Director** = script orchestration logic (budget loop, fan-out) + setup agent + final
  validation/arbitration agent.
- **TechLead** = agent: (1) analysis + roadmap; (2) each round returns, via a JSON schema, a
  "direction list + count + whether to stop", with each direction carrying specialty/category/focus_files
  to ensure orthogonality; maintains the insight blackboard and hypothesis ledger.
- **Engineer (specialist)** = parallel agents optimizing by specialty; plus dedicated
  benchmark/profile/verify/integrator agents.

## team_workflow.js flow
`args = {kernel_path (required), budget=6, gpu_ids="0", task, eval_dir, apply_to_original=false}`; the
`WORKFLOW_DIR` constant points at this directory. The script never touches the FS — agents do all of it.

1. **Setup** (director agent): the agent uses `date` to make a timestamp, creates
   `exp/team_<name>_<ts>/<name>/`, copies kernel→workspace+baseline, `git init`s the workspace and commits
   the baseline. → `{eval_dir, workspace, kernel_name, source_files}`
2. **Analyze + Roadmap** (tech_lead): analysis.json / codebase_context.md / roadmap.md. →
   `{kernel_type, kernel_file, modifiable_files, bottleneck_guess, roadmap_summary}`
3. **Benchmark setup** (benchmark_engineer): reuse/create task_runner, write COMMANDMENT.md, run 3× to
   check stability, record baseline. →
   `{correctness_cmd, benchmark_cmd, profile_cmd, baseline_per_case[], baseline_geomean_ms, reliable}`
4. **Baseline profile** (profile_engineer): profile_kernel.sh → baseline_metrics.json /
   profiling_summary.md. → `{bottleneck, key_metrics, top_opportunities}`
5. **Optimization loop** (JS while: dispatched<budget && noImprove<2):
   - a. **Plan round** (tech_lead): pass history (insight blackboard + hypothesis ledger) + profile +
     remaining, returns `{stop, directions:[{id,title,specialty,prompt,focus_files}]}`; the script clamps
     to remaining and round-robin-assigns gpu_id.
   - b/c. **pipeline(directions, optimize, verify)**: the specialist engineer optimizes on a private
     canonical copy and produces a patch → verify_engineer applies it on a clean copy and re-measures to
     produce **absolute** per-case latencies.
   - d. **integrate** (when ≥2 verified, off-budget): integrator merges/rewrites → re-measures.
   - e. Pick the winner (incl. integrate); absolute speedup = baseline_geomean_ms / candidate_geomean_ms;
     if winner > cumulative*1.05: the commit agent applies and commits it into canonical, writes
     current_best.diff; cumulative = winner_speedup; noImprove = 0; otherwise noImprove++.
   - f. On improvement, **re-profile** (profile_engineer).
   - dispatched += directions.length; the TechLead updates the insight blackboard / ledger and records
     history.
6. **Final report** (tech_lead): tech_lead_report.md (round-by-round, engineer-by-engineer + final
   per-case table + geomean/arith); write final_patch.diff (cumulative baseline→HEAD diff).
7. **Director validation + arbitration** (director agent): build a validation_workspace from the
   **original kernel_path**, apply final_patch, re-measure independently → director_validation.json; on a
   flag it may send back to the TechLead for one corrective round; write back per apply_to_original.

The script returns `{eval_dir, final_geomean, final_arithmetic, validation_status, report_path}`.

## budget semantics
Only "optimization-direction engineers" count; benchmark/profile/verify/integrate/commit/validate do not.
The script hard-clamps `directions ≤ budget-dispatched`. The TechLead may `stop` early.

## Generality
The script never branches on kernel type / single-kernel vs e2e; everything goes through the COMMANDMENT
discovered in the benchmark phase. vLLM/SGLang differ only in COMMANDMENT contents (start server /
throughput benchmark / output parity); the orchestration is unchanged.

## Validation steps (after building, run end-to-end to prove >50%)
1. Confirm the knn example HEAD is the pristine original; `git checkout` tracked files back to HEAD for a
   fair baseline; clean the build.
2. `rocm-smi` to pick an idle GPU.
3. Invoke via Workflow scriptPath, args={kernel_path=knn example, budget=6, gpu_ids=<idle>,
   apply_to_original=false}.
4. Read director_validation.json, confirm geomean > 54.8x. If not met, improve knowledge/roles/roadmap and
   re-run.
