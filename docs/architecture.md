# GEAK — Repository Architecture

MCP servers, agent hierarchy, and how they connect.

```mermaid
graph TB
    subgraph CLI ["CLI Layer"]
        GEAK_CLI["geak / geak-preprocess / geak-orchestrate"]
    end

    subgraph AGENTS ["Agent Hierarchy"]
        DA["<b>DefaultAgent</b>\n<i>Base: bash, editor, tools</i>"]
        DA --- SIA["StrategyInteractiveAgent\n<i>Strategy-based optimisation</i>"]
        DA --- SWE["SweAgent\n<i>Code-level modifications</i>"]
        DA --- PA["<b>ParallelAgent</b>\n<i>Wraps any agent class</i>"]
        DA --- SPA["SelectPatchAgent\n<i>LLM best-patch selection</i>"]
        DA --- OEW["OpenEvolveWorker\n<i>OpenEvolve task execution</i>"]
        DA --- UTA["UnitTestAgent\n<i>Test discovery / creation</i>"]

        PA --- SINGLE["single mode\n(num_parallel=1)\n1 agent + patch save"]
        PA --- PARALLEL["parallel mode\n(num_parallel>1)\nN agents on N GPUs\ngit worktrees"]
        PA --- POOL["pool mode\n(tasks=…)\nheterogeneous tasks\nacross GPU pool"]
    end

    subgraph MCP ["MCP Servers (mcp_tools/)"]
        MCP_ATD["<b>automated-test-discovery</b>\ndiscover"]
        MCP_PROF["<b>profiler-mcp</b>\nprofile_kernel\n(Metrix + rocprof-compute)"]
        MCP_METRIX["<b>metrix-mcp</b>\nprofile_kernel\n(AMD Metrix only)"]
        MCP_KP["<b>kernel-profiler</b>\nprofile_kernel\nbenchmark_kernel\nget_roofline_analysis\nget_bottleneck_suggestions"]
        MCP_KE["<b>kernel-evolve</b>\ngenerate_optimization\nmutate_kernel\ncrossover_kernels\nget_optimization_strategies\nsuggest_kernel_params"]
        MCP_ERCS["<b>kernel-ercs</b>\nevaluate_kernel_quality\nreflect_on_kernel_result\nget_amd_gpu_specs\ncheck_kernel_compatibility"]
        MCP_OE["<b>openevolve-mcp</b>\noptimize_kernel\ncheck_openevolve_status"]
    end

    subgraph BUILTIN ["Built-in Tools (src/minisweagent/tools/)"]
        T_RKU["resolve_kernel_url"]
        T_CMD["commandment"]
        T_VCMD["validate_commandment"]
        T_DISC["discovery"]
    end

    subgraph CTXPASS ["Context Passing (src/minisweagent/run/)"]
        CTX_GEN["codebase_context.py\ngenerate CODEBASE_CONTEXT.md"]
        CTX_DISP["dispatch.py\ninject context into\nsub-agent prompts"]
        CTX_TG["task_generator.py\ncontext available\nto LLM planner"]
        CTX_TR["ToolRuntime\nset_codebase_context()\npropagate to SubAgentTool"]
        CTX_GEN --> CTX_DISP --> CTX_TR
        CTX_GEN --> CTX_TG
    end

    subgraph MODELS ["LLM Backends (src/minisweagent/models/)"]
        M_AMD["amd_llm"]
        M_ANTH["anthropic"]
        M_LITE["litellm"]
    end

    CLI --> AGENTS
    GEAK_CLI -->|"preprocessor"| MCP_ATD & MCP_PROF & T_RKU & T_CMD & CTX_GEN
    SIA -->|"during optimisation"| MCP_KP & MCP_KE & MCP_ERCS
    OEW -->|"evolutionary search"| MCP_OE
    AGENTS -->|"LLM inference"| MODELS

    style CLI fill:#0c1a3d,stroke:#60a5fa,color:#e2e8f0
    style GEAK_CLI fill:#1e3a5f,color:#93c5fd,stroke:#60a5fa

    style AGENTS fill:#1e1b4b,stroke:#a78bfa,color:#e2e8f0
    style DA fill:#60a5fa,color:#000,stroke:#93c5fd
    style PA fill:#a78bfa,color:#000,stroke:#c4b5fd
    style SIA fill:#1e293b,color:#e2e8f0,stroke:#475569
    style SWE fill:#1e293b,color:#e2e8f0,stroke:#475569
    style SPA fill:#1e293b,color:#e2e8f0,stroke:#475569
    style OEW fill:#1e293b,color:#e2e8f0,stroke:#475569
    style UTA fill:#1e293b,color:#e2e8f0,stroke:#475569
    style SINGLE fill:#1e293b,color:#c4b5fd,stroke:#7c3aed
    style PARALLEL fill:#1e293b,color:#c4b5fd,stroke:#7c3aed
    style POOL fill:#1e293b,color:#c4b5fd,stroke:#7c3aed

    style MCP fill:#1c1917,stroke:#fbbf24,color:#fef3c7
    style MCP_ATD fill:#292524,color:#fde68a,stroke:#f59e0b
    style MCP_PROF fill:#292524,color:#fde68a,stroke:#f59e0b
    style MCP_METRIX fill:#292524,color:#fde68a,stroke:#f59e0b
    style MCP_KP fill:#292524,color:#fde68a,stroke:#f59e0b
    style MCP_KE fill:#292524,color:#fde68a,stroke:#f59e0b
    style MCP_ERCS fill:#292524,color:#fde68a,stroke:#f59e0b
    style MCP_OE fill:#292524,color:#fde68a,stroke:#f59e0b

    style BUILTIN fill:#022c22,stroke:#34d399,color:#d1fae5
    style T_RKU fill:#1e293b,color:#6ee7b7,stroke:#34d399
    style T_CMD fill:#1e293b,color:#6ee7b7,stroke:#34d399
    style T_VCMD fill:#1e293b,color:#6ee7b7,stroke:#34d399
    style T_DISC fill:#1e293b,color:#6ee7b7,stroke:#34d399

    style CTXPASS fill:#1a1a2e,stroke:#38bdf8,color:#bae6fd
    style CTX_GEN fill:#1e293b,color:#7dd3fc,stroke:#38bdf8
    style CTX_DISP fill:#1e293b,color:#7dd3fc,stroke:#38bdf8
    style CTX_TG fill:#1e293b,color:#7dd3fc,stroke:#38bdf8
    style CTX_TR fill:#1e293b,color:#7dd3fc,stroke:#38bdf8

    style MODELS fill:#2e1065,stroke:#c084fc,color:#f5d0fe
    style M_AMD fill:#1e293b,color:#e9d5ff,stroke:#a855f7
    style M_ANTH fill:#1e293b,color:#e9d5ff,stroke:#a855f7
    style M_LITE fill:#1e293b,color:#e9d5ff,stroke:#a855f7
```
