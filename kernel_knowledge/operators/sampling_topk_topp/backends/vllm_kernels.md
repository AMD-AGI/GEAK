---
title: sampling_topk_topp on vllm_kernels — SOTA card
kind: sota_card
operator: sampling_topk_topp
backend: vllm_kernels
gens: [gfx942, gfx950]
dtypes: [fp32, bf16]
regimes: [decode, both]
status: sota
updated: 2026-06-08
sources:
  - https://github.com/vllm-project/vllm/blob/main/vllm/v1/sample/ops/topk_topp_sampler.py
  - https://github.com/vllm-project/vllm/blob/main/vllm/v1/sample/sampler.py
  - https://github.com/vllm-project/vllm/blob/main/vllm/platforms/rocm.py
---

# sampling_topk_topp × vllm_kernels

## TL;DR
vLLM owns the **sampler orchestration** and on ROCm routes the hot path to **aiter**. `TopKTopPSampler`
has a dedicated `forward_hip` ("Optimized ROCm/aiter path") that calls `aiter.ops.sampling`
(`top_k_top_p_sampling_from_probs`, `top_p_sampling_from_probs`, `top_k_renorm_probs`); when aiter can't
serve (per-request generators, processed-logprobs, unsupported), it falls back to the **PyTorch sort**
(`apply_top_k_top_p_pytorch`). The no-sync random draw is vLLM's own **`random_sample`** (exponential/Gumbel
trick, *not* `torch.multinomial`). Greedy is the whole-batch **argmax fast path** in `Sampler.sample`.

## SOTA implementation(s)
| impl | source | gens/dtypes | measured perf | when best |
|---|---|---|---|---|
| `TopKTopPSampler.forward_hip` → `aiter_sample` | `vllm-project/vllm@HEAD:vllm/v1/sample/ops/topk_topp_sampler.py` | gfx942/950, fp32 | delegates to aiter rejection sampler (sorting-free, no sync) | the live MI300X sampling path |
| `random_sample` (Gumbel/exponential, no sync) | same | gfx942/950, fp32 | avoids `torch.multinomial` CPU-GPU sync | every random draw |
| `Sampler.sample` greedy argmax fast path | `vllm-project/vllm@HEAD:vllm/v1/sample/sampler.py` | gfx942/950 | whole-batch greedy → argmax, skip temp/top-k/top-p/min-p | temp=0 batches |
| `apply_top_k_top_p_pytorch` (sort) | topk_topp_sampler.py | any | O(V log V) sort, ~20% overhead; top-k-only has a GPU→CPU sync | fallback only |

## Config space / knobs
- `VLLM_ROCM_USE_AITER=1` → `rocm_aiter_ops.is_enabled()` true → `forward_hip` takes the aiter path.
- aiter branch selection: joint (`top_k_top_p_sampling_from_probs`), top-p only (`top_p_sampling_from_probs`),
  top-k only (`top_k_renorm_probs` + `torch.multinomial`).
- `--logprobs-mode` (raw/processed): processed-logprobs forces the native (sort) path (aiter can't return
  processed logits/logprobs).
- temperature/top_p must be **fp32** (#21936). Filter order: vLLM applies **top-k then top-p**.
- min-p is an **argmax-invariant** logits processor → applied only on the random branch.

## Numerics / parity
fp32; rejection path is **statistically** equivalent (KL gate, not token match); greedy is exact.
`random_sample` Gumbel draw on-device (no sync). Tie-break lowest index. aiter all-NaN/-inf → OOR id (guard
upstream). See [../numerics.md](../numerics.md).

## Integration (rebind seam)
Call site: `TopKTopPSampler.forward` dispatches to `forward_hip` on HIP. To force the native path for
debugging, disable aiter (`VLLM_ROCM_USE_AITER=0`) or hit a fallback trigger (per-request generator).
Verify: rocprofv3 shows the aiter sampling kernel (not a sort/`topk` Triton row) and **no D2H sync**
mid-decode; greedy parity after toggling.

## Pitfalls & anti-patterns
- Requesting **processed** logprobs → silently drops to the sort path (slow). Use raw logprobs for the
  fast path.
- Per-request `Generator` (seeded per request) → native fallback (aiter uses one philox seed).
- The top-k-only fallback (`apply_top_k_only`) carries a **GPU→CPU sync** — avoid on the hot path.
- bf16 temperature/top_p → crash/bias; keep fp32.
- Assuming `forward_cuda`'s FlashInfer behavior on AMD — AMD uses `forward_hip`/aiter (FlashInfer is
  NVIDIA-only).

## How to verify
rocprofv3: aiter sampling kernel present, no mid-decode `hipMemcpy D2H`; statistical KL gate vs sorted
multinomial; greedy exact parity; latency A/B vs `VLLM_ROCM_USE_AITER=0` (sort).

## Alternatives / cross-links
[aiter.md](aiter.md) (the kernels) · [hip.md](hip.md) · [triton.md](triton.md) · [../overview.md](../overview.md) ·
[[vllm_kernels]] · [[argmax_topk]] · [[lm_head_logits]].

## Sources
- `TopKTopPSampler` forward_hip/aiter_sample, sort fallback, `random_sample`, fp32 req: https://github.com/vllm-project/vllm/blob/main/vllm/v1/sample/ops/topk_topp_sampler.py
- `Sampler.sample` greedy fast path + argmax-invariant processors: https://github.com/vllm-project/vllm/blob/main/vllm/v1/sample/sampler.py
- ROCm aiter enablement: https://github.com/vllm-project/vllm/blob/main/vllm/platforms/rocm.py
