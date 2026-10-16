---
kind: landscape
updated: 2026-10-16
title: Kernel-agent research references
scope: Systems that automatically write/optimize GPU kernels (CUDA + Triton + ROCm/HIP + NPU)
audience: perf_knowledge — single-kernel optimization ladder, GEAK, e2e_workflow
---

# Kernel-agent research references

Use these projects for search/evaluation ideas and benchmark artifacts. Published
scores differ in workloads, baselines, hardware, budgets and failure handling;
they are not direct GEAK versus GEAK-EVO comparisons.

| Reference | Useful question |
|---|---|
| [KernelBench](https://github.com/ScalingIntelligence/KernelBench) | How are tasks, reference outputs and timing defined? |
| [robust-kbench](https://github.com/SakanaAI/robust-kbench) | Which generated-kernel shortcuts can evade weak correctness checks? |
| [TritonBench](https://github.com/thunlp/TritonBench) | Which Triton tasks and dependency assumptions apply? |
| [GEAK paper](https://arxiv.org/abs/2507.23194) | How do generation, feedback and optimization interact? |
| [KernelEvolve](https://arxiv.org/abs/2512.23236) | How can retained candidates support continuing search across hardware? |
| [Kevin-32B](https://arxiv.org/abs/2507.11948) | What does training against execution feedback establish? |
| [GPU MODE KernelBot](https://github.com/gpu-mode/kernelbot) | How are submissions and benchmark records retained? |
| [KernelGYM](https://github.com/hkust-nlp/KernelGYM) | What reusable execution/evaluation interfaces are available? |

Useful design questions include whether a measured improvement survives full-call
validation, whether a proposed intervention actually executed, and whether useful
alternatives remain available after a failed or slower attempt. Test these against
the current task instead of copying a reported success rate or fixed search depth.

Correctness covers the declared numerical/state contract, including changing inputs
and outputs, aliases and warmed paths when applicable. Restrictions on library calls
or languages come from that task, not a benchmark's unrelated policy. Keep reference,
correctness and timing attribution explicit; no single gate is “reward-hacking-proof.”

## Sources

- Papers and project repositories linked above.
- [GEAK agents-family report](https://rocm.blogs.amd.com/artificial-intelligence/geak-agents-family/README.html)
- [Rigorous agentic benchmark practices](https://arxiv.org/pdf/2507.02825)
