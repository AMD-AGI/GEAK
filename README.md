# PerfSkills

A collection of [Claude Code](https://docs.anthropic.com/en/docs/claude-code) skills for GPU kernel performance optimization.

## Available Skills

| Skill | Target Hardware | Description |
|-------|----------------|-------------|
| [GEAK](skills/geak/SKILL.md) | AMD MI300X (gfx942) | Autonomous GPU kernel optimization with parallel workers |

## GEAK: GPU Expert Agent for Kernel Optimization

GEAK is a 6-phase autonomous pipeline that analyzes, profiles, and optimizes GPU kernels (Triton or HIP) on AMD MI300X hardware. It spawns parallel optimization workers, each exploring a different strategy, and selects the best verified result.

### Pipeline

```
Phase 1: Analyze       Understand kernel type, dependencies, hardware
              |
Phase 2: Test Harness   Create/discover tests, generate evaluation contract
              |
Phase 3: Profile        Profile with rocprof-compute, classify bottleneck
              |
Phase 4: Plan           Generate diverse optimization strategies
              |
Phase 5: Optimize       Spawn N parallel workers on separate GPUs
              |
Phase 6: Evaluate       Verify results, select best, generate report
```

### Features

- **Parallel optimization**: Multiple workers explore different strategies simultaneously, each on a dedicated GPU
- **Automatic profiling**: Bottleneck classification (memory-bound, compute-bound, latency-bound, LDS-bound) drives strategy selection
- **Multi-round iteration**: Up to N rounds of optimization, each building on the previous best result
- **Correctness verification**: Every optimization is validated against a reference before acceptance
- **Structured output**: Per-test-case results with geometric and arithmetic mean speedups

### Supported Kernel Types

| Type | Detection | Examples |
|------|-----------|---------|
| Triton | `.py` files with `@triton.jit` or `tl.` | Fused attention, custom GEMM, reduction kernels |
| HIP | `.hip`, `.cu`, `.cpp` files with `__global__` | Hand-written CUDA/HIP kernels, ported CUDA code |

## Prerequisites

- **Hardware**: AMD MI300X GPU (gfx942)
- **Software**:
  - ROCm 6.x
  - `hipcc` compiler
  - `rocprof-compute` (profiling)
  - Python 3.8+
  - [Claude Code](https://docs.anthropic.com/en/docs/claude-code) CLI

## Quick Start

### 1. Install the skill

Copy or symlink the `skills/geak` directory into your Claude Code skills location:

```bash
# Option A: Copy to project-local skills
mkdir -p .claude/skills
cp -r /path/to/PerfSkills/skills/geak .claude/skills/

# Option B: Symlink
ln -s /path/to/PerfSkills/skills/geak .claude/skills/geak
```

### 2. Run optimization

In Claude Code, invoke the skill:

```
/geak --kernel_path /path/to/kernel.hip --repo_path /path/to/repo
```

With optional parameters:

```
/geak --kernel_path /path/to/kernel.py \
      --repo_path /path/to/repo \
      --num_parallel 4 \
      --gpu_ids 0,1,2,3 \
      --max_rounds 3
```

### 3. Check results

All outputs are saved under `./kernel_eval/<kernel_name>_<timestamp>/`:

```
kernel_eval/<kernel_name>_<YYYYMMDD_HHMMSS>/
├── baseline/           # Original kernel + baseline metrics
├── optimized/          # Best optimized kernel + patch
├── logs/               # Pipeline artifacts + per-worker logs
└── report/
    ├── final_report.json   # Machine-readable results
    └── summary.md          # Human-readable summary
```

## Benchmark Results

Tested on 13 HIP kernels on AMD MI300X. Speedup = arithmetic mean across test cases per kernel.

| Kernel | GEAK_v3 | Claude Code GEAK | Winner |
|--------|---------|------------------|--------|
| roipoint_pool3d | 16.82x | 14.61x | GEAK_v3 |
| ball_query | 11.62x | 13.14x | Claude Code |
| roiaware_pool3d | 10.24x | 9.92x | GEAK_v3 |
| three_nn | 1.43x | 8.82x | Claude Code |
| knn | FAIL | 6.56x | Claude Code |
| assign_score_withk | 3.76x | 4.00x | Claude Code |
| silu | 1.21x | 1.26x | Claude Code |
| matrix_multiplication | 1.14x | 1.19x | Claude Code |
| three_interpolate | 1.01x | 1.15x | Claude Code |
| furthest_point_sample | FAIL | 1.04x | Claude Code |
| points_in_boxes | 1.03x | 1.04x | Tie |
| gather_points | 1.32x | 0.96x | GEAK_v3 |
| mla_decode | N/A | 586.00x | Claude Code |

| Summary | GEAK_v3 | Claude Code |
|---------|---------|-------------|
| Wins | 3 | 9 |
| Failures | 2 | 0 |
| Arith Mean (12 common, fail=1.0x) | 4.30x | 5.31x |

## Repository Structure

```
PerfSkills/
├── README.md
├── LICENSE
├── skills/
│   └── geak/
│       ├── SKILL.md                  # Main orchestrator (6-phase pipeline)
│       ├── knowledge/                # Domain knowledge base
│       │   ├── amd_mi300x_guide.md   # MI300X hardware reference
│       │   ├── hip_patterns.md       # HIP optimization patterns
│       │   ├── triton_patterns.md    # Triton optimization patterns
│       │   ├── optimization_strategies.md  # Per-bottleneck strategies
│       │   ├── profiling_analysis.md # Profiling interpretation guide
│       │   └── working_memory_guide.md    # Worker self-monitoring
│       ├── scripts/                  # Tooling
│       │   ├── create_harness.py     # Test harness generator
│       │   └── profile_kernel.sh     # Profiling wrapper
│       └── sub_skills/               # Phase instructions
│           ├── analyze.md            # Phase 1: Kernel analysis
│           ├── test_harness.md       # Phase 2: Test setup
│           ├── profile.md            # Phase 3: Profiling
│           ├── plan.md              # Phase 4: Strategy planning
│           ├── optimize_worker.md    # Phase 5: Worker instructions
│           └── evaluate.md           # Phase 6: Result evaluation
```

## License

Apache License 2.0. See [LICENSE](LICENSE) for details.
