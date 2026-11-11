---
title: Operator libraries and dispatch
kind: landscape
updated: 2026-10-16
scope: operator libraries and integration references
---

# Operator libraries and dispatch

Use libraries to find correct baselines and reusable implementations. Check the
actual build's shape, layout, dtype, workspace and graph-capture support before
calling a candidate. Package-level AMD support does not establish every operator's
coverage or the implementation selected at runtime.

| Library | Useful entry point or distinction |
|---|---|
| [aiter](https://github.com/ROCm/aiter) | AMD operators with CK, assembly, Triton and FlyDSL variants; inspect the actual callable and tuned dispatch |
| [FlagGems](https://github.com/FlagOpen/FlagGems) | Triton operator implementations and vendor dispatch; useful candidate source |
| [PyTorch dispatcher](https://docs.pytorch.org/tutorials/advanced/extend_dispatcher) | Dispatch keys select implementations; `torch.compile` may fuse or replace an eager call |
| [Liger-Kernel](https://github.com/linkedin/Liger-Kernel) | Fused training operations; validate backward and numerical contracts when required |
| [FlashInfer](https://github.com/flashinfer-ai/flashinfer) / [ROCm fork](https://github.com/ROCm/flashinfer) | Attention/KV primitives; upstream and ROCm fork coverage differ by revision |
| [xFormers](https://github.com/facebookresearch/xformers) / [ROCm fork](https://github.com/ROCm/xformers) | Attention implementations with operator-specific constraints |
| [HF kernels](https://huggingface.co/docs/kernels/index) | Versioned prebuilt kernels; match Torch/ROCm ABI and pin repository revision |
| [BackendBench](https://github.com/meta-pytorch/BackendBench) | Operator comparison and correctness tooling; adapt coverage to the task |

For vLLM/SGLang integration, use [serving dispatch](serving_stack_registries.md).
For per-shape configurations, use [autotuning](autotuning_frameworks.md). Preserve
fallback behavior and verify which path actually ran; changing a registry entry
or importing a package alone does not demonstrate workload engagement.

## Sources

- Official repositories and documentation linked in the table.
- [BackendBench correctness](https://github.com/meta-pytorch/BackendBench/blob/main/docs/correctness.md)
- [PyTorch custom operators](https://docs.pytorch.org/docs/stable/compile/programming_model.custom_ops.html)
