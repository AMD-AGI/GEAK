---
title: Serving-stack kernel selection
kind: landscape
updated: 2026-10-16
---

# Serving-stack kernel selection

A useful integration point identifies the operation, selected backend/variant,
configuration and actual callable. Verify this path under the measured workload;
a config flag, heuristic choice or matching kernel name alone is insufficient.

| Stack | Mechanism to inspect | Version-sensitive detail |
|---|---|---|
| vLLM | `CustomOp` platform methods, attention backend selection, compiled operator calls | Names and aiter flags/configuration evolve; use installed help and source |
| SGLang | `ATTENTION_BACKENDS` factories and prefill/decode backend overrides | Defaults depend on device/model; distinct phases may use a hybrid dispatcher |
| HF `kernels` | `register_kernel_mapping`, `kernelize`, versioned repository handles | Device, mode and compiler support can select fallback; pin the resolved build |
| PyTorch | `torch.library` custom ops and compiler lowerings/patterns | An opaque custom op restricts fusion; eager selection may differ after compilation |
| TensorRT-LLM | Plugins and cached autotuner tactics | CUDA-oriented design reference, not an AMD implementation recipe |

The [vLLM `ir_op_priority` proposal](https://github.com/vllm-project/vllm/issues/33163)
is design history, not proof that those flags exist in the installed version.
Likewise, a static capability mapping chooses a compatible candidate; it does not
establish that candidate as fastest. Reuse matching tuned records and time the
actual operation and workload after integration.

When adding a backend, preserve the public signature, state/alias semantics and
fallback behavior. Check imports, generated bindings and graph-capture behavior.
A model-level throughput change needs a matching workload comparison; isolated
kernel timing alone cannot attribute it.

## Sources

- [vLLM custom operations](https://docs.vllm.ai/en/latest/design/custom_op/)
- [SGLang attention registry](https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/layers/attention/attention_registry.py)
- [SGLang backend options](https://docs.sglang.ai/advanced_features/attention_backend.html)
- [HF layer mappings](https://huggingface.co/docs/kernels/en/layers)
- [PyTorch custom operations](https://docs.pytorch.org/docs/stable/compile/programming_model.custom_ops.html)
- [TensorRT-LLM](https://nvidia.github.io/TensorRT-LLM/)
