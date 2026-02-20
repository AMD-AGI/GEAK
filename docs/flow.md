# GEAK — End-to-End Optimization Flow

```mermaid
flowchart TB
    START(["<b>Input:</b> kernel URL + GPU IDs"]) --> P1

    subgraph PREPROCESS ["Phase 1 — Preprocessing (geak-preprocess)"]
        direction TB
        P1["resolve-kernel-url\n→ resolved.json"]
        P1b["codebase-context\n→ CODEBASE_CONTEXT.md"]
        P2["test-discovery (MCP)\n→ discovery.json"]
        P3["kernel-profile (MCP)\n→ profile.json"]
        P4["baseline-metrics\n→ baseline_metrics.json"]
        P5["commandment\n→ COMMANDMENT.md"]
        P1 --> P1b --> P2 --> P3 --> P4 --> P5
    end

    P5 --> ORCH_START

    subgraph ORCH ["Phase 2 — Orchestration Loop (geak-orchestrate)"]
        direction TB
        ORCH_START["Orchestrator LLM Agent\n<i>reads preprocessor artefacts +\nCODEBASE_CONTEXT.md</i>"]
        GEN["<b>generate_tasks</b>\nLLM creates task .md files\nwith num_gpus per task\n(fusion, vectorisation, coalescing, OpenEvolve …)"]
        DISPATCH["<b>dispatch_tasks</b>\nParallelAgent pool mode\nassigns tasks → GPU(s) via worktrees\n<i>multi-GPU tasks get comma-sep HIP_VISIBLE_DEVICES</i>"]
        ORCH_START --> GEN --> DISPATCH

        DISPATCH --> GPU0["GPU 0\n<i>Strategy Agent</i>"]
        DISPATCH --> GPU1["GPU 1\n<i>Strategy Agent</i>"]
        DISPATCH --> GPU23["GPU 2,3\n<i>OpenEvolve Worker</i>\n(num_gpus=2)"]
        DISPATCH --> GPU4["GPU 4\n<i>SWE Agent</i>"]

        COLLECT["<b>collect_results</b>\nRead back per-task results\nValidate against COMMANDMENT"]
        GPU0 & GPU1 & GPU23 & GPU4 --> COLLECT

        DECIDE{{"Improvement found?"}}
        COLLECT --> DECIDE
        DECIDE -->|"Yes → iterate"| GEN
        DECIDE -->|"No → stop"| FINAL["<b>finalize</b>\n→ final_report.json"]
    end

    FINAL --> RESULT(["<b>Output:</b> Best patch + report\ngeak_output/"])

    style START fill:#fbbf24,color:#000,stroke:#fcd34d
    style RESULT fill:#34d399,color:#000,stroke:#6ee7b7
    style PREPROCESS fill:#1e1b4b,stroke:#a78bfa,color:#e2e8f0
    style ORCH fill:#0c1a3d,stroke:#60a5fa,color:#e2e8f0
    style DECIDE fill:#fbbf24,color:#000,stroke:#fcd34d
    style P1 fill:#1e293b,color:#e2e8f0,stroke:#475569
    style P1b fill:#1e293b,color:#e2e8f0,stroke:#475569
    style P2 fill:#1e293b,color:#e2e8f0,stroke:#475569
    style P3 fill:#1e293b,color:#e2e8f0,stroke:#475569
    style P4 fill:#1e293b,color:#e2e8f0,stroke:#475569
    style P5 fill:#1e293b,color:#e2e8f0,stroke:#475569
    style ORCH_START fill:#1e293b,color:#e2e8f0,stroke:#60a5fa
    style GEN fill:#1e293b,color:#e2e8f0,stroke:#475569
    style DISPATCH fill:#1e293b,color:#e2e8f0,stroke:#475569
    style GPU0 fill:#164e63,color:#a5f3fc,stroke:#22d3ee
    style GPU1 fill:#164e63,color:#a5f3fc,stroke:#22d3ee
    style GPU23 fill:#4a1d6a,color:#e9d5ff,stroke:#c084fc
    style GPU4 fill:#164e63,color:#a5f3fc,stroke:#22d3ee
    style COLLECT fill:#1e293b,color:#e2e8f0,stroke:#475569
    style FINAL fill:#1e293b,color:#e2e8f0,stroke:#60a5fa
```
