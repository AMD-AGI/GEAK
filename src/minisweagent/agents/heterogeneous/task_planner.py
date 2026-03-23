"""Dynamic task planner -- generates M optimization tasks from discovery results.

Given a DiscoveryResult (with kernel info, dependency graph, and fusion
opportunities), generates a prioritized list of AgentTask objects. The number
of tasks is determined by what optimizations make sense for the kernel, NOT
by the number of available GPUs. The GPU pool scheduler handles the mapping.
"""

from __future__ import annotations

from minisweagent.agents.agent_spec import AgentTask
from minisweagent.run.preprocess.discovery_types import DiscoveryResult

_GPU_AND_PROFILER_RULES = """
## GPU and Profiler Rules (CRITICAL -- read carefully)

1. **HIP_VISIBLE_DEVICES is ALREADY SET** in your environment by the scheduler.
   Do NOT prefix commands with `HIP_VISIBLE_DEVICES=X`. Do NOT set or export it.
   It is already correct. Adding it inline will CRASH rocprofv3.

2. **profile_kernel tool**: Pass ONLY the python command, e.g.:
   `python3 /path/to/harness.py --profile`
   Do NOT prefix with env vars -- rocprofv3 uses os.execvpe(), not a shell.

3. **COMMANDMENT.md** (for OpenEvolve) MUST use EXACTLY these section headers:
   `## SETUP`, `## CORRECTNESS`, `## PROFILE`
   Any other header is SILENTLY IGNORED. Commands must NOT start with `cd`,
   `source`, `export`, or any shell built-in.

4. **Use absolute paths** in all commands. Do not use `cd /path && ...`.
"""

_BUILD_CONTEXT = {
    "python": "Triton kernels are JIT-compiled. No build step needed. Edit .py files directly.",
    "cpp": (
        "HIP/CK kernels require compilation with hipcc/nvcc.\nAfter editing .cu/.cpp files, rebuild before testing."
    ),
    "asm": "HSACO assembly is precompiled. Only the Python wrapper and launch configuration can be modified.",
}


def build_optimization_tasks(
    discovery_result: DiscoveryResult,
    base_task_context: str,
    agent_class: type,
) -> list[AgentTask]:
    """Generate all optimization tasks from discovery results."""
    if not discovery_result.kernels:
        return []

    kernel = discovery_result.kernels[0]
    dep_graph = discovery_result.dependency_graphs.get(kernel.kernel_name)
    tasks: list[AgentTask] = []

    ktype = kernel.kernel_type
    lang = kernel.kernel_language
    build_ctx = _BUILD_CONTEXT.get(lang, "") + _GPU_AND_PROFILER_RULES
    inner = kernel.inner_kernel_path
    wrapper = kernel.file_path

    if ktype == "triton":
        tasks.extend(_triton_tasks(agent_class, base_task_context, build_ctx, kernel, inner, wrapper))
    elif ktype == "hip":
        tasks.extend(_hip_tasks(agent_class, base_task_context, build_ctx, kernel, inner, wrapper))
    elif ktype == "ck":
        tasks.extend(_ck_tasks(agent_class, base_task_context, build_ctx, kernel, wrapper))
    elif ktype == "asm":
        tasks.extend(_asm_tasks(agent_class, base_task_context, build_ctx, kernel, wrapper))
    else:
        tasks.extend(_generic_tasks(agent_class, base_task_context, build_ctx, kernel, inner, wrapper))

    if dep_graph:
        for i, opp in enumerate(dep_graph.fusion_opportunities):
            target_lang = _pick_fusion_target_lang(opp.languages)
            tasks.append(
                AgentTask(
                    agent_class=agent_class,
                    task=(
                        f"{base_task_context}\n\n"
                        f"{build_ctx}\n\n"
                        f"## Kernel Fusion Task\n"
                        f"{opp.description}\n\n"
                        f"Fusion type: {opp.fusion_type}\n"
                        f"Involved nodes: {', '.join(opp.involved_nodes)}\n"
                        f"Languages: {opp.languages}\n"
                        f"Target language for fused kernel: {target_lang}\n\n"
                        f"Dependency graph:\n{dep_graph.summary()}\n\n"
                        "Fuse the identified operations to eliminate intermediate "
                        "memory round-trips and reduce kernel launch overhead."
                    ),
                    label=f"fusion-{i}",
                    priority=5,
                    kernel_language=target_lang,
                )
            )

    tasks.append(
        AgentTask(
            agent_class=agent_class,
            task=(
                f"{base_task_context}\n\n"
                f"{build_ctx}\n\n"
                "## Profile-Guided Optimization\n"
                "Profile the kernel using the profiler MCP tool. Identify the "
                "top performance bottleneck and implement a targeted fix."
            ),
            label="profile-guided",
            priority=15,
            kernel_language=lang,
        )
    )

    return sorted(tasks, key=lambda t: t.priority)


def _triton_tasks(agent_class, ctx, build_ctx, kernel, inner, wrapper) -> list[AgentTask]:
    tasks = []
    target = inner or wrapper

    if inner:
        tasks.append(
            AgentTask(
                agent_class=agent_class,
                task=(
                    f"{ctx}\n\n{build_ctx}\n\n"
                    "## OpenEvolve on Inner Kernel\n"
                    f"Run OpenEvolve on the inner Triton kernel at {inner}.\n"
                    "Follow the INSTRUCTIONS.md workflow: create COMMANDMENT.md "
                    "and baseline_metrics.json, then run OpenEvolve.\n"
                    f"Wrapper file: {wrapper}\n\n"
                    "COMMANDMENT.md MUST have EXACTLY these 3 sections:\n"
                    "  ## SETUP\n  ## CORRECTNESS\n  ## PROFILE\n"
                    "NO other section headers. Commands must use ABSOLUTE PATHS.\n"
                    "Do NOT use cd, source, export as command prefixes.\n"
                    "Do NOT prefix with HIP_VISIBLE_DEVICES (already set).\n"
                    "Use ${GEAK_WORK_DIR} and ${GEAK_GPU_DEVICE} variables.\n"
                    "Create a wrapper shell script in SETUP that sets env vars."
                ),
                label="openevolve-inner",
                priority=0,
                kernel_language="python",
            )
        )

    tasks.append(
        AgentTask(
            agent_class=agent_class,
            task=(
                f"{ctx}\n\n{build_ctx}\n\n"
                "## Triton Autotune Configuration\n"
                f"Optimize Triton autotuning configs for {target}: "
                "BLOCK_M, BLOCK_N, BLOCK_K, num_warps, num_stages, "
                "waves_per_eu. Try expanding the autotune search space "
                "or adding new configurations."
            ),
            label="triton-autotune",
            priority=8,
            kernel_language="python",
        )
    )

    tasks.append(
        AgentTask(
            agent_class=agent_class,
            task=(
                f"{ctx}\n\n{build_ctx}\n\n"
                "## Algorithmic Memory Optimization\n"
                f"Optimize memory access patterns in {target}: "
                "improve coalescing, use shared memory (tl.load with "
                "eviction_policy), optimize tiling, reduce bank conflicts, "
                "use vectorized loads where possible."
            ),
            label="triton-algorithmic",
            priority=1,
            kernel_language="python",
        )
    )

    return tasks


def _hip_tasks(agent_class, ctx, build_ctx, kernel, inner, wrapper) -> list[AgentTask]:
    tasks = []
    target = inner or wrapper

    tasks.append(
        AgentTask(
            agent_class=agent_class,
            task=(
                f"{ctx}\n\n{build_ctx}\n\n"
                "## OpenEvolve on HIP Kernel\n"
                f"Run OpenEvolve on the HIP kernel at {target}.\n"
                "Follow the INSTRUCTIONS.md workflow: create COMMANDMENT.md "
                "and baseline_metrics.json, then run OpenEvolve.\n"
                f"Wrapper file: {wrapper}\n\n"
                "Use ${GEAK_WORK_DIR} and ${GEAK_GPU_DEVICE} variables.\n"
                "Create a wrapper shell script in SETUP that sets env vars."
            ),
            label="openevolve-hip",
            priority=0,
            kernel_language="cpp",
        )
    )

    tasks.append(
        AgentTask(
            agent_class=agent_class,
            task=(
                f"{ctx}\n\n{build_ctx}\n\n"
                "## HIP Launch Configuration\n"
                f"Optimize the HIP kernel launch configuration for {target}: "
                "block size, grid size, shared memory allocation. "
                "Target maximum occupancy using the occupancy calculator."
            ),
            label="hip-launch-config",
            priority=15,
            kernel_language="cpp",
        )
    )

    tasks.append(
        AgentTask(
            agent_class=agent_class,
            task=(
                f"{ctx}\n\n{build_ctx}\n\n"
                "## HIP Memory Optimization\n"
                f"Optimize HIP kernel memory access for {target}: "
                "coalescing, LDS usage, vectorized loads (float4/half8), "
                "minimize bank conflicts, use __ldg for read-only data."
            ),
            label="hip-memory",
            priority=5,
            kernel_language="cpp",
        )
    )

    return tasks


def _ck_tasks(agent_class, ctx, build_ctx, kernel, wrapper) -> list[AgentTask]:
    tasks = []

    tasks.append(
        AgentTask(
            agent_class=agent_class,
            task=(
                f"{ctx}\n\n{build_ctx}\n\n"
                "## OpenEvolve on CK Kernel\n"
                f"Run OpenEvolve on the Composable Kernel at {wrapper}.\n"
                "Follow the INSTRUCTIONS.md workflow: create COMMANDMENT.md "
                "and baseline_metrics.json, then run OpenEvolve.\n\n"
                "Use ${GEAK_WORK_DIR} and ${GEAK_GPU_DEVICE} variables.\n"
                "Create a wrapper shell script in SETUP that sets env vars."
            ),
            label="openevolve-ck",
            priority=0,
            kernel_language="cpp",
        )
    )

    tasks.append(
        AgentTask(
            agent_class=agent_class,
            task=(
                f"{ctx}\n\n{build_ctx}\n\n"
                "## CK Template Parameter Tuning\n"
                f"Tune Composable Kernel template parameters for {wrapper}: "
                "tile sizes (MPerBlock, NPerBlock, KPerBlock), pipeline depth, "
                "vector widths. Requires hipcc rebuild after changes."
            ),
            label="ck-template-tuning",
            priority=8,
            kernel_language="cpp",
        )
    )

    tasks.append(
        AgentTask(
            agent_class=agent_class,
            task=(
                f"{ctx}\n\n{build_ctx}\n\n"
                "## CK Pipeline Exploration\n"
                f"Explore alternative CK tile operations or pipeline "
                f"configurations for {wrapper}."
            ),
            label="ck-pipeline",
            priority=5,
            kernel_language="cpp",
        )
    )

    return tasks


def _asm_tasks(agent_class, ctx, build_ctx, kernel, wrapper) -> list[AgentTask]:
    return [
        AgentTask(
            agent_class=agent_class,
            task=(
                f"{ctx}\n\n{build_ctx}\n\n"
                "## ASM Kernel Wrapper Optimization\n"
                "This kernel uses precompiled HSACO assembly. The binary "
                "itself cannot be modified. Optimize the launch configuration "
                f"and the Python wrapper around it at {wrapper}."
            ),
            label="asm-launch-config",
            priority=15,
            kernel_language="asm",
        )
    ]


def _generic_tasks(agent_class, ctx, build_ctx, kernel, inner, wrapper) -> list[AgentTask]:
    target = inner or wrapper
    return [
        AgentTask(
            agent_class=agent_class,
            task=(
                f"{ctx}\n\n{build_ctx}\n\n"
                "## General Kernel Optimization\n"
                f"Optimize {target} for maximum performance. "
                "Profile first, then apply targeted improvements."
            ),
            label="general-optimization",
            priority=5,
            kernel_language=kernel.kernel_language,
        ),
    ]


def _pick_fusion_target_lang(languages: set[str]) -> str:
    """Choose the target language for a fused kernel."""
    if "triton" in languages and "asm" not in languages:
        return "python"
    if "ck" in languages:
        return "cpp"
    return "cpp"
