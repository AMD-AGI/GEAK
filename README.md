# PerfSkills

GPU kernel optimization skills for LLM-based coding agents. Target: AMD MI300X (gfx942).

## Skills

| Skill | Architecture | Iteration | Key Idea |
|-------|-------------|-----------|----------|
| [GEAK](skills/geak/SKILL.md) | Orchestrator + parallel workers | Single round | One agent plans, N workers execute in parallel |
| [Team](skills/team/SKILL.md) | Director → TechLead → Engineers | Multi-round with re-profiling | Hierarchical delegation, budget-controlled iteration, wrapper overhead detection |

### Design Comparison

**GEAK** is a flat pipeline: analyze → profile → plan → spawn N workers → evaluate. Each worker independently optimizes the kernel, and the best result wins. Simple and effective for kernels where a single optimization direction suffices.

**Team** adds depth: a Director validates results independently, a TechLead iterates across rounds (re-profiling after each to detect bottleneck shifts), and Engineers execute diverse strategies that a Merge Engineer can combine. It also detects when Python/C++ wrapper overhead dominates kernel compute and redirects optimization accordingly. Better for complex kernels where compounding optimizations across rounds matters.

## Quick Start

```
/wekafs/zihao/2026/geak_cc/PerfSkills/skills/team/SKILL.md \
  kernel_path=/wekafs/zihao/2026/geak_cc/PerfSkills/examples/tasks/knn/ \
  budget=6 \
  gpu_ids=0 \
  task="Optimize this kernel for maximum throughput on AMD MI300X"
```

## Results

12 HIP kernels on AMD MI300X (arithmetic mean speedup):

| Method | Mean Speedup | Best Kernel |
|--------|-------------|-------------|
| GEAK | 5.26x | ball_query 12.64x |
| Team | **9.19x** | knn 25.50x |

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
