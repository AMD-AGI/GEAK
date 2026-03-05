# GEAK — Repository Architecture

MCP servers, agent hierarchy, and how they connect.

```mermaid
graph TB
    subgraph CLI ["CLI Layer"]
        GEAK_CLI["geak / geak-preprocess / geak-orchestrate"]
    end

    subgraph AGENTS ["Agent Hierarchy"]
        DA["<b>DefaultAgent</b>\n<i>Base: bash, editor, tools</i>"]
        DA --- IA["InteractiveAgent\n<i>Human/confirm/yolo modes</i>"]
        IA --- SA["StrategyAgent\n<i>Strategy management + UI</i>"]
        SA --- SIA["StrategyInteractiveAgent\n<i>Rich console strategy agent</i>"]
        IA --- SWE["SweAgent\n<i>Code-level modifications</i>"]
        DA --- PA["<b>ParallelAgent</b>\n<i>Wraps any agent class</i>"]
        DA --- SPA["SelectPatchAgent\n<i>LLM best-patch selection</i>"]
        DA --- OEW["OpenEvolveWorker\n<i>OpenEvolve task execution</i>"]
        DA --- UTA["UnitTestAgent\n<i>Test discovery / creation</i>"]

        PA --- SINGLE["single mode\n(num_parallel=1)\n1 agent + patch save"]
        PA --- HOMOGENEOUS["homogeneous mode\n(default)\nN identical agents, 1 GPU each"]
        PA --- HETEROGENEOUS["heterogeneous mode\n(agent_specs=…)\ndifferent agent types\nfixed GPU assignments"]
        PA --- POOL["pool mode\n(tasks=…)\nM tasks on N GPUs\ntasks queue as GPUs free"]
    end

    subgraph MCP ["MCP Servers (mcp_tools/)"]
        MCP_ATD["<b>automated-test-discovery</b>\ndiscover"]
        MCP_PROF["<b>profiler-mcp</b>\nprofile_kernel\n(Metrix + rocprof-compute)"]
        MCP_METRIX["<b>metrix-mcp</b>\nprofile_kernel\n(AMD Metrix only)"]
        MCP_KE["<b>kernel-evolve</b>\ngenerate_optimization\nmutate_kernel\ncrossover_kernels\nget_optimization_strategies\nsuggest_kernel_params"]
        MCP_ERCS["<b>kernel-ercs</b>\nevaluate_kernel_quality\nreflect_on_kernel_result\nget_amd_gpu_specs\ncheck_kernel_compatibility"]
        MCP_OE["<b>openevolve-mcp</b>\noptimize_kernel\ncheck_openevolve_status"]
    end

    subgraph BUILTIN ["Built-in Tools (src/minisweagent/tools/)"]
        T_BASH["bash"]
        T_EDIT["str_replace_editor"]
        T_SAT["save_and_test"]
        T_SUB["submit"]
        T_STRAT["strategy_manager"]
        T_BL["baseline_metrics"]
        T_RKU["resolve_kernel_url"]
        T_CMD["commandment"]
        T_VCMD["validate_commandment"]
        T_COMPAT["check_kernel_compatibility"]
        T_SUBAG["sub_agent"]
    end

    subgraph SHARED ["Shared Pipeline Helpers (src/minisweagent/run/)"]
        PH["<b>pipeline_helpers.py</b>\nload_geak_model / geak_model_factory\nadd_agent_filter_args / apply_agent_filter_env\nextract_harness_path / validate_harness\nexecute_harness_validation\ncreate_validated_harness\ninject_pipeline_context\nrun_baseline_profile\nREQUIRED_HARNESS_FLAGS\nMAX_HARNESS_RETRIES\nDEFAULT_AGENT_BENCHMARK_ITERATIONS\nDEFAULT_EVAL_BENCHMARK_ITERATIONS"]
        DT["<b>discovery_types.py</b>\n<i>(src/minisweagent/tools/)</i>\nDiscoveryResult.from_dict()"]
    end

    subgraph CTXPASS ["Context Passing (src/minisweagent/run/)"]
        CTX_GEN["codebase_context.py\ngenerate CODEBASE_CONTEXT.md"]
        CTX_DISP["dispatch.py\nrun_task_batch()\ntask_file_to_agent_task()\ninject context into\nsub-agent prompts"]
        CTX_TG["task_generator.py\ngenerate_tasks()\ncontext available\nto LLM planner"]
        CTX_TR["ToolRuntime\nset_codebase_context()\nset_env() / set_cwd()\npropagate to SubAgentTool"]
        CTX_GEN --> CTX_DISP --> CTX_TR
        CTX_GEN --> CTX_TG
    end

    subgraph MODELS ["LLM Backends (src/minisweagent/models/)"]
        M_AMD["amd_llm"]
        M_ANTH["anthropic"]
        M_LITE["litellm"]
        M_OR["openrouter"]
        M_PK["portkey"]
    end

    CLI --> AGENTS
    GEAK_CLI -->|"preprocessor"| MCP_ATD & MCP_PROF & T_RKU & T_CMD & CTX_GEN
    GEAK_CLI -->|"shared helpers"| PH & DT
    CTX_DISP -->|"context injection"| PH
    SIA -->|"during optimisation"| MCP_KE & MCP_ERCS
    OEW -->|"evolutionary search"| MCP_OE
    AGENTS -->|"LLM inference"| MODELS
```
