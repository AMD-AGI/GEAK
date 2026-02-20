# GEAK — User-Facing Tools

```mermaid
block-beta
    columns 5

    geak["<b>geak</b>\nFull pipeline or single-task mode\n--gpu-ids 0,1,2,3\n--allowed-agents / --excluded-agents"]:5

    space:5

    geak_preprocess["<b>geak-preprocess</b>\nPreprocessing pipeline"]:2
    space
    geak_orchestrate["<b>geak-orchestrate</b>\nOrchestration loop\n--gpu-ids 0,1,2,3"]:2

    space:5

    resolve["resolve-kernel-url\nGitHub URL → local\ncheckout"]
    context["codebase-context\nRepo layout →\nCODEBASE_CONTEXT.md"]
    discover["test-discovery\nFind tests &\nbenchmarks (MCP)"]
    profile["kernel-profile\nGPU profiling\n(Metrix + rocprof)"]
    baseline["baseline-metrics\nDuration, throughput,\nbottleneck"]

    space:5

    space:2
    commandment["commandment\nGenerate COMMANDMENT.md"]:1
    space:2

    space:5

    space
    task_gen["task-generator\nLLM-driven task creation\n--num-gpus N"]
    run_tasks["run-tasks\nBatch task execution\n(ParallelAgent pool)"]:2
    select_patch["select-patch\nLLM-based\nbest patch selection"]

    space:5

    space
    strategy["strategy_agent\nLLM + kernel-evolve\nMCP tools"]
    swe["swe_agent\nManual edits,\nautotune configs"]
    openevolve["openevolve-worker\nOpenEvolve\noptimisation"]
    space

    space:5

    standalone["<b>Standalone utilities</b>"]:5

    space:5

    validate["validate-commandment\nValidate COMMANDMENT.md"]:2
    space
    geak_task["geak --from-task\nSingle task sub-agent\n(strategy / swe)\n--gpu-ids N"]:2

    space:5

    shared["<b>pipeline_helpers.py</b>\nShared module used by all tools above:\nload_geak_model, inject_pipeline_context,\nvalidate_harness, create_validated_harness,\nrun_baseline_profile, agent filter helpers,\nDiscoveryResult.from_dict()"]:5

    geak --> geak_preprocess
    geak --> geak_orchestrate
    geak_preprocess --> resolve
    geak_preprocess --> context
    geak_preprocess --> discover
    geak_preprocess --> profile
    geak_preprocess --> baseline
    geak_preprocess --> commandment
    geak_orchestrate --> task_gen
    geak_orchestrate --> run_tasks
    geak_orchestrate --> select_patch
    run_tasks --> strategy
    run_tasks --> swe
    run_tasks --> openevolve
    geak_preprocess --> shared
    geak_orchestrate --> shared
    run_tasks --> shared
    task_gen --> shared
```
