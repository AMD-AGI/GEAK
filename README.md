# PerfSkills

GPU kernel optimization skills for LLM-based coding agents. Target: AMD MI300X (gfx942).

## Skills


| Skill                        | Architecture                    | Iteration                     | Key Idea                                                                         |
| ---------------------------- | ------------------------------- | ----------------------------- | -------------------------------------------------------------------------------- |
| [GEAK](skills/geak/SKILL.md) | Orchestrator + parallel workers | Single round                  | One agent plans, N workers execute in parallel                                   |
| [Team](skills/team/SKILL.md) | Director → TechLead → Engineers | Multi-round with re-profiling | Hierarchical delegation, budget-controlled iteration, wrapper overhead detection |


### Design Comparison


|                        | GEAK                                       | Team                                                      |
| ---------------------- | ------------------------------------------ | --------------------------------------------------------- |
| **Origin**             | Refactored from GEAK_v3, reuses its logic  | Ground-up redesign                                        |
| **Architecture**       | Flat: Orchestrator + parallel workers      | Hierarchical: Director → TechLead → Engineers → Merge     |
| **Iteration**          | Single round                               | Multi-round, budget-controlled                            |
| **Re-profiling**       | No                                         | Yes, with bottleneck shift analysis                       |
| **Patch combination**  | Best-of-N                                  | Merge Engineer combines top patches per round             |
| **Wrapper detection**  | No                                         | Auto-detects host overhead, redirects optimization        |
| **Validation**         | Orchestrator verifies                      | Director independently re-benchmarks                      |
| **Best for**           | Single optimization direction suffices     | Complex kernels needing compounded multi-round gains      |


**GEAK** is a flat pipeline: analyze → profile → plan → spawn N workers → evaluate. Each worker independently optimizes the kernel, and the best result wins.

**Team** adds depth: a Director validates results independently, a TechLead iterates across rounds (re-profiling after each to detect bottleneck shifts), and Engineers execute diverse strategies that a Merge Engineer can combine. It also detects when Python/C++ wrapper overhead dominates kernel compute and redirects optimization accordingly.

## Quick Start

```
skills/team/SKILL.md \
  kernel_path=examples/tasks/knn/ \
  budget=6 \
  gpu_ids=0 \
  task="Optimize this kernel for maximum throughput on AMD MI300X"
```

## Results

12 HIP kernels on AMD MI300X:

| Method | Arith Mean | Wins |
| ------ | ---------- | ---- |
| GEAK_v3 | 4.30x | 0 |
| GEAK Skill | 5.26x | 3 |
| Team Skill | **9.19x** | **9** |


Per-kernel breakdown: [examples/result/hip2hip_comparison.md](examples/result/hip2hip_comparison.md)

## Repository Structure

```
PerfSkills/
├── skills/
│   ├── geak/              # Single-round parallel optimization
│   │   ├── SKILL.md
│   │   ├── knowledge/
│   │   ├── scripts/
│   │   └── sub_skills/
│   └── team/              # Multi-round hierarchical optimization
│       ├── SKILL.md
│       ├── knowledge/     # 7 knowledge files (MI300X, HIP, Triton, strategies, profiling, wrapper, self-monitoring)
│       ├── scripts/
│       └── sub_skills/    # analyze, benchmark_setup, profile, engineer, evaluate, merge_engineer, tech_lead
├── examples/
│   ├── tasks/             # Example kernel tasks
│   │   └── knn/           # K-Nearest Neighbors kernel
│   └── result/            # Benchmark comparisons
└── exp/                   # Experiment outputs
```

## Prerequisites

- AMD MI300X GPU (gfx942), ROCm 6.x, `rocprof-compute`, Python 3.8+

## License

Apache License 2.0