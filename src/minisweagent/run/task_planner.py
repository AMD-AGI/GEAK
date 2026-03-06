# Copyright(C) [2026] Advanced Micro Devices, Inc. All rights reserved. Portions of this file consist of AI-generated content.
# SPDX-License-Identifier: Apache-2.0

"""Dynamic task planner -- generates M optimization tasks from discovery results.

Given a DiscoveryResult (with kernel info, dependency graph, and fusion
opportunities), generates a prioritized list of AgentTask objects. The number
of tasks is determined by what optimizations make sense for the kernel, NOT
by the number of available GPUs. The GPU pool scheduler in ParallelAgent
handles the mapping of M tasks to N GPU slots.

Priority scheme (lower = higher priority, runs first):
  0  -- OpenEvolve on the inner kernel (highest impact, automated)
  5  -- Kernel fusion opportunities (one task per detected opportunity)
  5  -- Language-specific advanced tuning (CK templates, HIP occupancy)
  10 -- Wrapper / launch parameter tuning
  10 -- Algorithmic memory optimization
  15 -- Profile-guided (generic fallback)
"""

from __future__ import annotations

from minisweagent.agents.agent_spec import AgentTask
from minisweagent.tools.discovery_types import DiscoveryResult

# Rules injected into EVERY task prompt to prevent common agent mistakes.
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

# Build context strings per language for inclusion in task prompts
_BUILD_CONTEXT = {
    "python": "Triton kernels are JIT-compiled. No build step needed. Edit .py files directly.",
    "cpp": (
        "HIP/CK kernels require compilation with hipcc/nvcc.\nAfter editing .cu/.cpp files, rebuild before testing."
    ),
    "asm": ("HSACO assembly is precompiled. Only the Python wrapper and launch configuration can be modified."),
}


def build_optimization_tasks(
    discovery_result: DiscoveryResult,
    base_task_context: str,
    agent_class: type,
) -> list[AgentTask]:
    """Generate all optimization tasks from discovery results.

    Args:
        discovery_result: Output of DiscoveryPipeline.run().
        base_task_context: Common context string (kernel paths, discovered
            tests/benchmarks, INSTRUCTIONS.md reference) prepended to each task.
        agent_class: The agent class to use for all tasks (typically StrategyAgent).

    Returns:
        List of AgentTask sorted by priority. May contain more tasks than
        available GPUs -- the pool scheduler handles queuing.
    """
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

    # ----------------------------------------------------------------
    # Language-specific strategies
    # ----------------------------------------------------------------

    if ktype == "triton":
        tasks.extend(
            _triton_tasks(
                agent_class,
                base_task_context,
                build_ctx,
                kernel,
                inner,
                wrapper,
            )
        )
    elif ktype == "hip":
        tasks.extend(
            _hip_tasks(
                agent_class,
                base_task_context,
                build_ctx,
                kernel,
                inner,
                wrapper,
            )
        )
    elif ktype == "ck":
        tasks.extend(
            _ck_tasks(
                agent_class,
                base_task_context,
                build_ctx,
                kernel,
                wrapper,
            )
        )
    elif ktype == "asm":
        tasks.extend(
            _asm_tasks(
                agent_class,
                base_task_context,
                build_ctx,
                kernel,
                wrapper,
            )
        )
    else:
        # Unknown type -- fall back to generic strategies
        tasks.extend(
            _generic_tasks(
                agent_class,
                base_task_context,
                build_ctx,
                kernel,
                inner,
                wrapper,
            )
        )

    # ----------------------------------------------------------------
    # Cross-cutting: fusion opportunities (one task per opportunity)
    # ----------------------------------------------------------------
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
                        f"Fuse the identified operations to eliminate intermediate "
                        f"memory round-trips and reduce kernel launch overhead."
                    ),
                    label=f"fusion-{i}",
                    priority=5,
                    kernel_language=target_lang,
                )
            )

    # ----------------------------------------------------------------
    # Cross-cutting: profile-guided (always applicable)
    # ----------------------------------------------------------------
    tasks.append(
        AgentTask(
            agent_class=agent_class,
            task=(
                f"{base_task_context}\n\n"
                f"{build_ctx}\n\n"
                f"## Profile-Guided Optimization\n"
                f"Profile the kernel using the profiler MCP tool. Identify the "
                f"top performance bottleneck (memory bandwidth, occupancy, "
                f"instruction throughput, etc.) and implement a targeted fix."
            ),
            label="profile-guided",
            priority=15,
            kernel_language=lang,
        )
    )

    return sorted(tasks, key=lambda t: t.priority)


# ====================================================================
# Per-language task generators
# ====================================================================


def _triton_tasks(
    agent_class,
    ctx,
    build_ctx,
    kernel,
    inner,
    wrapper,
) -> list[AgentTask]:
    tasks = []
    target = inner or wrapper

    # OpenEvolve on the inner kernel
    if inner:
        tasks.append(
            AgentTask(
                agent_class=agent_class,
                task=(
                    f"{ctx}\n\n{build_ctx}\n\n"
                    f"## OpenEvolve on Inner Kernel\n"
                    f"Run OpenEvolve on the inner Triton kernel at {inner}.\n"
                    f"Follow the INSTRUCTIONS.md workflow: create COMMANDMENT.md "
                    f"and baseline_metrics.json, then run OpenEvolve.\n"
                    f"Wrapper file: {wrapper}\n\n"
                    f"COMMANDMENT.md MUST have EXACTLY these 3 sections:\n"
                    f"  ## SETUP\n  ## CORRECTNESS\n  ## PROFILE\n"
                    f"NO other section headers. Commands must use ABSOLUTE PATHS.\n"
                    f"Do NOT use cd, source, export as command prefixes.\n"
                    f"Do NOT prefix with HIP_VISIBLE_DEVICES (already set).\n"
                    f"Use ${{GEAK_WORK_DIR}} and ${{GEAK_GPU_DEVICE}} variables.\n"
                    f"Create a wrapper shell script in SETUP that sets env vars."
                ),
                label="openevolve-inner",
                priority=0,
                kernel_language="python",
            )
        )

    # Autotune config exploration
    tasks.append(
        AgentTask(
            agent_class=agent_class,
            task=(
                f"{ctx}\n\n{build_ctx}\n\n"
                f"## Triton Autotune Configuration\n"
                f"Optimize Triton autotuning configs for {target}: "
                f"BLOCK_M, BLOCK_N, BLOCK_K, num_warps, num_stages, "
                f"waves_per_eu. Try expanding the autotune search space "
                f"or adding new configurations."
            ),
            label="triton-autotune",
            priority=10,
            kernel_language="python",
        )
    )

    # Algorithmic memory optimization
    tasks.append(
        AgentTask(
            agent_class=agent_class,
            task=(
                f"{ctx}\n\n{build_ctx}\n\n"
                f"## Algorithmic Memory Optimization\n"
                f"Optimize memory access patterns in {target}: "
                f"improve coalescing, use shared memory (tl.load with "
                f"eviction_policy), optimize tiling, reduce bank conflicts, "
                f"use vectorized loads where possible."
            ),
            label="triton-algorithmic",
            priority=10,
            kernel_language="python",
        )
    )

    return tasks


def _hip_tasks(
    agent_class,
    ctx,
    build_ctx,
    kernel,
    inner,
    wrapper,
) -> list[AgentTask]:
    tasks = []
    target = inner or wrapper

    tasks.append(
        AgentTask(
            agent_class=agent_class,
            task=(
                f"{ctx}\n\n{build_ctx}\n\n"
                f"## OpenEvolve on HIP Kernel\n"
                f"Run OpenEvolve on the HIP kernel at {target}.\n"
                f"Follow the INSTRUCTIONS.md workflow: create COMMANDMENT.md "
                f"and baseline_metrics.json, then run OpenEvolve.\n"
                f"Wrapper file: {wrapper}\n\n"
                f"Use ${{GEAK_WORK_DIR}} and ${{GEAK_GPU_DEVICE}} variables.\n"
                f"Create a wrapper shell script in SETUP that sets env vars."
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
                f"## HIP Launch Configuration\n"
                f"Optimize the HIP kernel launch configuration for {target}: "
                f"block size, grid size, shared memory allocation. "
                f"Target maximum occupancy using the occupancy calculator."
            ),
            label="hip-launch-config",
            priority=5,
            kernel_language="cpp",
        )
    )

    tasks.append(
        AgentTask(
            agent_class=agent_class,
            task=(
                f"{ctx}\n\n{build_ctx}\n\n"
                f"## HIP Memory Optimization\n"
                f"Optimize HIP kernel memory access for {target}: "
                f"coalescing, LDS usage, vectorized loads (float4/half8), "
                f"minimize bank conflicts, use __ldg for read-only data."
            ),
            label="hip-memory",
            priority=10,
            kernel_language="cpp",
        )
    )

    return tasks


def _ck_tasks(
    agent_class,
    ctx,
    build_ctx,
    kernel,
    wrapper,
) -> list[AgentTask]:
    tasks = []

    tasks.append(
        AgentTask(
            agent_class=agent_class,
            task=(
                f"{ctx}\n\n{build_ctx}\n\n"
                f"## OpenEvolve on CK Kernel\n"
                f"Run OpenEvolve on the Composable Kernel at {wrapper}.\n"
                f"Follow the INSTRUCTIONS.md workflow: create COMMANDMENT.md "
                f"and baseline_metrics.json, then run OpenEvolve.\n\n"
                f"Use ${{GEAK_WORK_DIR}} and ${{GEAK_GPU_DEVICE}} variables.\n"
                f"Create a wrapper shell script in SETUP that sets env vars."
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
                f"## CK Template Parameter Tuning\n"
                f"Tune Composable Kernel template parameters for {wrapper}: "
                f"tile sizes (MPerBlock, NPerBlock, KPerBlock), pipeline depth, "
                f"vector widths. Requires hipcc rebuild after changes."
            ),
            label="ck-template-tuning",
            priority=5,
            kernel_language="cpp",
        )
    )

    tasks.append(
        AgentTask(
            agent_class=agent_class,
            task=(
                f"{ctx}\n\n{build_ctx}\n\n"
                f"## CK Pipeline Exploration\n"
                f"Explore alternative CK tile operations or pipeline "
                f"configurations for {wrapper}."
            ),
            label="ck-pipeline",
            priority=10,
            kernel_language="cpp",
        )
    )

    return tasks


def _asm_tasks(
    agent_class,
    ctx,
    build_ctx,
    kernel,
    wrapper,
) -> list[AgentTask]:
    return [
        AgentTask(
            agent_class=agent_class,
            task=(
                f"{ctx}\n\n{build_ctx}\n\n"
                f"## ASM Kernel Wrapper Optimization\n"
                f"This kernel uses precompiled HSACO assembly. The binary "
                f"itself cannot be modified. Optimize the launch configuration "
                f"(grid/block dims, shared memory) and the Python wrapper "
                f"around it at {wrapper}."
            ),
            label="asm-launch-config",
            priority=10,
            kernel_language="asm",
        )
    ]


def _generic_tasks(
    agent_class,
    ctx,
    build_ctx,
    kernel,
    inner,
    wrapper,
) -> list[AgentTask]:
    target = inner or wrapper
    return [
        AgentTask(
            agent_class=agent_class,
            task=(
                f"{ctx}\n\n{build_ctx}\n\n"
                f"## General Kernel Optimization\n"
                f"Optimize {target} for maximum performance. "
                f"Profile first, then apply targeted improvements."
            ),
            label="general-optimization",
            priority=10,
            kernel_language=kernel.kernel_language,
        ),
    ]


def _pick_fusion_target_lang(languages: set[str]) -> str:
    """Choose the target language for a fused kernel."""
    if "triton" in languages and "asm" not in languages:
        return "python"
    elif "ck" in languages:
        return "cpp"
    else:
        return "cpp"
