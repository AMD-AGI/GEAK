---
kind: landscape
updated: 2026-10-16
hardware: [MI300X (gfx942, CDNA3), MI325X (gfx942, CDNA3), MI350X/MI355X (gfx950, CDNA4)]
scope: published AMD implementation examples by operator
sourcing: numbers tagged vendor/third-party + hardware + version; see ## Sources
---

# Published AMD kernel examples

This file retains the historical survey path. These are candidate implementations
and published optimization examples, not a current per-operator ranking. Check
the cited source's shape, dtype, product, software revision and measurement boundary
before transferring a result. This cleanup did not rerun its benchmarks.

| Operation | Implementation examples to inspect |
|---|---|
| Dense/scaled GEMM | hipBLASLt, CK, aiter variants; [Gluon tutorial](https://rocm.blogs.amd.com/software-tools-optimization/gluon-gemm-tutorial/README.html), [CDNA4 HIP GEMM](https://rocm.blogs.amd.com/software-tools-optimization/cdna4-gemm-kernels/README.html) |
| Attention / MLA | aiter, CK, Triton, TileLang; [aiter MLA](https://rocm.blogs.amd.com/software-tools-optimization/aiter-mla/README.html) |
| Fused MoE | aiter CK/assembly/FlyDSL paths; [Kimi FlyDSL example](https://rocm.blogs.amd.com/artificial-intelligence/kimi-k2.5-optimize/README.html) |
| Norm, activation, RoPE, quantization | aiter and serving compiler fusions; [vLLM fusion passes](https://docs.vllm.ai/en/latest/design/fusions/) |
| Communication | RCCL, MoRI and task-specific collectives; [QuickReduce example](https://rocm.blogs.amd.com/artificial-intelligence/quick-reduce/README.html) |

[HipKittens](https://github.com/HazyResearch/HipKittens) also provides explicit
AMD scheduling examples. Ping-pong/interleaving, low-precision paths and fused
communication are hypotheses to test, not universally preferred replacements.
Preserve the task's numerical contract, topology and full-operation timing.

Use [operator cards](../operators/) to locate call sites and supported configurations,
and [roofline guidance](../profiling/roofline_on_mi.md) to interpret percentages.
A tutorial's MFMA-efficiency number is not automatically the fraction of a device's
datasheet peak. Library throughput and system-level benchmark scores measure
different boundaries; neither supplies a universal software ceiling.

## Sources

- Official implementations and reports linked in the table.
- [aiter releases](https://github.com/ROCm/aiter/releases)
- [HipKittens paper](https://arxiv.org/abs/2511.08083)
