# team_workflow — Dynamic Workflow for Kernel / Model Inference Optimization

A deterministic **Workflow** (JS-orchestrated multi-agent pipeline) that optimizes the inference
speed of a GPU kernel directory — a single kernel, several kernels fused together, or an end-to-end
vLLM / SGLang model — on AMD MI300X. It is the dynamic-workflow successor to the markdown-driven
`skills/team` skill: the budget loop, round fan-out, and verification are now **JS control flow**,
while every judgement call is made by an agent returning **structured JSON**.

## What's different from the old `team` skill (and why it's better)
1. **Deterministic orchestration** — the budget loop / parallelism / verification live in
   `team_workflow.js`, not in LLM-interpreted prose. The TechLead returns structured decisions.
2. **Independent verification of every claimed speedup** — each engineer's patch is re-benchmarked
   by a separate `verify_engineer` in a clean workspace *as soon as it finishes* (pipelined). The
   script trusts only verified, absolute-latency numbers → the winner is genuinely the fastest.
3. **Specialist engineers (A)** — `algorithm | memory | compute | host_runtime`; each loads only its
   relevant knowledge → focused context, sharper results, naturally orthogonal & mergeable.
4. **Host/Runtime as a first-class track (B)** — attacks the wall-clock floor (dispatch collapse,
   native layouts, CUDA graph, wrapper overhead). This is where the last 1.5–3x of geomean lives.
5. **Cross-round memory (C)** — an insight blackboard + hypothesis ledger threads what was learned
   into the next round's engineer prompts; dead-ends are not retried.
6. **Integrator (E)** — combines the round's winning ideas (stack compatible patches OR hand-merge
   conflicting ones into a coherent best implementation). Does not consume budget.
7. **Director arbitration (H)** — independently validates the final patch against the TRUE original
   baseline and can flag / request a corrective round.

## Roles → workflow mapping
- **Director** = the script's orchestration + a setup agent + a final validation/arbitration agent.
- **TechLead** = agent for analyze/roadmap, per-round planning (orthogonal directions + stop), the
  cross-round memory, and the final report.
- **Engineers** = parallel specialist agents (optimize), plus `benchmark_engineer`, `profile_engineer`,
  `verify_engineer`, and `integrator`.

## Pipeline
`Setup → Analyze+Roadmap → Benchmark(COMMANDMENT+baseline) → Baseline Profile →`
`LOOP[ Plan round → (Optimize ‖ Verify, pipelined) → Integrate → Commit winner → Re-profile → Update memory ] →`
`Final Report → Director Validation`.

Each round's winner is committed into the canonical workspace, so the next round builds on the
cumulative best. Speedup is always measured in **absolute latency vs the true baseline**:
`geomean( baseline_ms / optimized_ms )`.

## Budget
`budget` = the **total number of optimization directions** the TechLead may dispatch to engineers
across all rounds. Only optimization-direction engineers count; benchmark / profile / verify /
integrate / commit / validate do **not** consume budget. The script hard-caps each round to the
remaining budget; the TechLead may also stop early (`stop=true`) when further directions won't pay.
Example (budget=6): round 1 = 3 directions, round 2 = 3; or 4 then 2; or stop after 4.

## Invocation
This is a Workflow, run via the `Workflow` tool with `scriptPath` and `args`. **No paths are
hard-coded in the script** — it is portable to any install location. Set `scriptPath` to wherever
this folder lives and pass that same folder as `args.workflow_dir`:

```
Workflow({
  scriptPath: "<WF_DIR>/team_workflow.js",   // <WF_DIR> = absolute path to THIS workflows/ folder
  args: {
    kernel_path: "/abs/path/to/kernel_or_model_dir",  // REQUIRED
    workflow_dir: "<WF_DIR>",  // REQUIRED: same folder as scriptPath (holds roles/ knowledge/ scripts/);
                               //           a JS workflow can't read its own path, so the caller passes it
    budget: 6,                 // optional, default 6
    gpu_ids: "0",              // optional, comma-separated, default "0"
    task: "focus on ...",      // optional natural-language steer
    exp_root: "",              // optional, output root; default = sibling "exp/" next to workflow_dir
    eval_dir: "",              // optional, override the output dir for this single run
    apply_to_original: "false",// optional; if "true", write the validated patch back to kernel_path
    // --- author mode (write a fresh implementation from scratch, then optimize it) ---
    mode: "optimize",          // optional: "optimize" (default, edit an existing kernel) | "author"
    target_language: "triton", // author mode: triton (always) | hip | ck — the language to write
    op_spec: {},               // author mode: {op_kind, shapes, dtype, math_contract, regime} for the op
    kernel_knowledge_dir: ""   // optional: AMD authoring knowledge base the author_engineer reads
  }
})
```

### Author mode (NEW)
`mode="author"` is for when there is **no existing source to optimize** — a hot op (e.g. a library
GEMM/attention) needs a fresh implementation. Here `kernel_path` is an **op task dir** holding the
IMMUTABLE oracle (`meta.json` + `unittest.py` + optional `reference_io.pt`). The `author_engineer`
writes the simplest correct implementation in `target_language` (correctness-judged against the
oracle), commits it as the baseline, and then the **same optimize loop** improves it. Returns
`authored:false` / `validation_status:"author_failed"` if no correct baseline can be produced (the
caller drops that language). `mode="optimize"` (default) is unchanged and fully backward compatible.

`<WF_DIR>` is the only location-specific value and it is supplied at call time (it is just the
dirname of `scriptPath`). Everything else is derived: `exp_root` defaults to `<parent of WF_DIR>/exp`.

The user-facing prompt stays minimal & generic, e.g.:
- `优化下 /xxx/xxx/knn`
- `优化下 /xxx/xxx/knn， budget 6，重点优化下 wrapper 开销`
These map to `kernel_path` (+ optional `budget` / `task`). No repo URL needed.

## Output
Everything lands under `<exp_root>/team_<kernel>_<timestamp>/<kernel>/` (default `exp_root` =
the `exp/` folder sibling to `workflow_dir`):
- `COMMANDMENT.md`, `baseline_timing.json`, `analysis.json`, `codebase_context.md`, `roadmap.md`
- `baseline_metrics.json`, `profiling_summary.md`
- `round_N/engineer_i/{worker_result.json, report.md, best_patch.diff}` — each engineer's mini-report
- `round_N/integrate/`, `insight_log.md`, `current_best.diff`
- `tech_lead_report.md` — round-by-round narrative + final per-case table (the TechLead summary)
- `final_patch.diff`, `optimized/`, `director_validation.json` — the official verified result

## Generality (single kernel ↔ e2e model)
The script never branches on kernel type or single-vs-e2e. Everything flows through the
**COMMANDMENT** discovered/built at the Benchmark phase (setup / correctness / benchmark / profile
commands + a parse hint). For a vLLM/SGLang model the only difference is what those commands contain
(launch the server, run a throughput/latency benchmark, define output-parity correctness); the
Director/TechLead/Engineer orchestration is identical.

## Files
```
team_workflow.js     orchestration (deterministic)
roles/               director, tech_lead, engineer, benchmark_engineer,
                     profile_engineer, verify_engineer, integrator
knowledge/           optimization_strategies, hip/triton/wrapper, profiling_guide,
                     amd_mi300x, self_monitoring, geomean_levers
scripts/             gpu_lock.sh, profile_kernel.sh
```
