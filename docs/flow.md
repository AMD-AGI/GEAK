# GEAK — End-to-End Optimization Flow

```mermaid
flowchart TB
    START(["<b>Input:</b> kernel URL + GPU IDs"]) --> P1

    subgraph PREPROCESS ["Phase 1 — Preprocessing (geak-preprocess)"]
        direction TB
        P1["Step 1/7: resolve-kernel-url\n→ resolved.json"]
        P1b["Step 2/7: codebase-context\n→ CODEBASE_CONTEXT.md"]
        P2["Step 3/7: test-discovery (MCP)\n→ discovery.json"]
        P2b["Step 3b/3c: create-validated-harness\n(UnitTestAgent + static/runtime\nvalidation + retry loop)\n→ test_command, harness_results.json\n<i>via pipeline_helpers.py</i>"]
        P2c["Baseline benchmarks\n(re-run all modes with eval iterations)\n→ benchmark_baseline.txt\n→ full_benchmark_baseline.txt"]
        P3["Step 5/7: kernel-profile (MCP)\n→ profile.json\n<i>with warmup</i>"]
        P4["Step 6/7: baseline-metrics\n→ baseline_metrics.json\n<i>(enriched with benchmark_duration_us)</i>"]
        P5["Step 7/7: commandment\n→ COMMANDMENT.md"]
        P1 --> P1b --> P2 --> P2b --> P2c --> P3 --> P4 --> P5
    end

    P5 --> MODE_CHOICE

    MODE_CHOICE{{"--heterogeneous?"}}
    MODE_CHOICE -->|"No (default)"| HOMO_START
    MODE_CHOICE -->|"Yes"| HETERO_START

    subgraph HOMO ["Phase 2a — Homogeneous Orchestration"]
        direction TB
        HOMO_START["All agents get the\nsame task each round"]
        HOMO_TASK["Write single task file\n(00_optimize.md)"]
        HOMO_DISPATCH["run_task_batch\nN copies → N GPUs"]
        HOMO_EVAL["_evaluate_round_best\nFULL_BENCHMARK + PROFILE\non best kernel"]
        HOMO_EARLY{{"Improved over\nprior best?"}}
        HOMO_START --> HOMO_TASK --> HOMO_DISPATCH --> HOMO_EVAL --> HOMO_EARLY
        HOMO_EARLY -->|"Yes → next round\n(accumulate starting_patch)"| HOMO_TASK
        HOMO_EARLY -->|"No → early stop"| HOMO_FINAL
        HOMO_FINAL["_auto_finalize\n→ final_report.json"]
    end

    subgraph HETERO ["Phase 2b — Heterogeneous Orchestration (LLM-driven)"]
        direction TB
        HETERO_START["Orchestrator LLM Agent\n<i>reads preprocessor artefacts +\nCODEBASE_CONTEXT.md +\nmemory context</i>"]
        HETERO_EXPLORE["Phase 1: Exploration\nLLM reads kernel, profiling,\nCOMMANDMENT via bash/editor"]
        GEN["<b>generate_tasks</b>\nLLM creates task .md files\nwith num_gpus per task\n(fusion, vectorisation, coalescing, OpenEvolve …)"]
        DISPATCH["<b>dispatch_tasks</b>\nParallelAgent pool mode\nassigns tasks → GPU(s) via worktrees\n<i>inject_pipeline_context()\nensures all agents get identical context</i>"]
        HETERO_START --> HETERO_EXPLORE --> GEN --> DISPATCH

        DISPATCH --> GPU0["GPU 0\n<i>Strategy Agent</i>"]
        DISPATCH --> GPU1["GPU 1\n<i>Strategy Agent</i>"]
        DISPATCH --> GPU23["GPU 2,3\n<i>OpenEvolve Worker</i>\n(num_gpus=2)"]
        DISPATCH --> GPU4["GPU 4\n<i>SWE Agent</i>"]

        COLLECT["<b>collect_results</b>\nRead back per-task results\nValidate against COMMANDMENT"]
        GPU0 & GPU1 & GPU23 & GPU4 --> COLLECT

        EVAL_ROUND["<b>_evaluate_round_best</b>\nCreate eval worktree\nApply best patch\nRun FULL_BENCHMARK + PROFILE\nVerified speedup comparison"]
        COLLECT --> EVAL_ROUND

        DECIDE{{"More rounds?"}}
        EVAL_ROUND --> DECIDE
        DECIDE -->|"Yes → iterate\n(feed eval into next round)"| GEN
        DECIDE -->|"Final round"| FINAL["<b>finalize</b>\n→ final_report.json"]
        DECIDE -->|"Step limit hit"| AUTO_FINAL["<b>_auto_finalize</b>\nAuto-select best across\nall rounds"]
    end

    HOMO_FINAL --> RESULT(["<b>Output:</b> Best patch + report\ngeak_output/"])
    FINAL --> RESULT
    AUTO_FINAL --> RESULT
```
