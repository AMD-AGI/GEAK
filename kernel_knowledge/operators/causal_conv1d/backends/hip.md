---
title: causal_conv1d on HIP/C++ — SOTA card
kind: sota_card
operator: causal_conv1d
backend: hip
gens: [gfx942, gfx950]
dtypes: [bf16, fp16, fp32]
regimes: [decode]
status: sota
updated: 2026-06-05
sources:
  - ROCm/aiter@a6bb499375849eec45d68c5ccaebc8865fd422c0:csrc/kernels/causal_conv1d_update.cu
  - ROCm/aiter@a6bb499375849eec45d68c5ccaebc8865fd422c0:aiter/ops/causal_conv1d.py
  - ROCm/aiter@a6bb499375849eec45d68c5ccaebc8865fd422c0:csrc/include/causal_conv1d.h
---

# causal_conv1d × HIP/C++

## TL;DR
A hand-written HIP kernel (`causal_conv1d_update`) for the **decode** step: one wavefront (64 threads)
per batch, channels split across lanes, weights + the `width`-tap sliding window held **entirely in
registers**, ≤4-tap MAC in fp32, optional SiLU, in-place `conv_state` update (linear shift or circular
buffer). It is the textbook right shape for a launch-bound decode op — no LDS, no barriers, minimal VGPR,
fully specialized on `width` and buffer mode at compile time. Use it as the single-step decode path; for
varlen prefill use the Triton `causal_conv1d_fn` ([triton.md](triton.md)), which the HIP file does not
cover (it is `update`-only).

## SOTA implementation(s)
| impl | source | gens/dtypes/shapes | measured perf | when best |
|---|---|---|---|---|
| `causal_conv1d_update` HIP kernel | `ROCm/aiter@a6bb49937:csrc/kernels/causal_conv1d_update.cu::causal_conv1d_update_kernel` | gfx942/950 (comment: tuned for MI308 64-lane wave); fp16/bf16/fp32 (itype×wtype independent); width 2/3/4; `[batch,dim,seqlen]` | launch/state-I/O bound; comparable to the Triton update (~tens of µs) | single-/few-token decode, continuous batching, circular state |

JIT-compiled via aiter (`@compile_ops("module_causal_conv1d_update")`, `aiter/ops/causal_conv1d.py`);
first call pays a one-time build, then hits the cached `.so`.

## Config space / knobs
- `kNThreads = 64` — exactly one wavefront (CDNA wave64); one block per batch, `blockIdx.y*64+tid` →
  channel. Grid `(batch, ceil(dim/64))`.
- `kWidth` (2/3/4) and `kIsCircularBuffer` are **template params** → the tap loop (`#pragma unroll`) and
  the state-update branch are fully specialized; no runtime branch in the hot loop. Host dispatches width
  via `causal_conv1d_update_dispatch`.
- `weight_vals[kWidth]`, `x_vals[kWidth]` in registers; `__launch_bounds__(kNThreads)`. No LDS, no
  `__syncthreads`. This is the whole tuning story — there is nothing to sweep.
- Circular mode keyed on `cache_seqlens != nullptr` (`update_idx = cache_seqlen % state_len`); else linear
  shift-left.

## Numerics / parity
fp32 accumulate (`float weight_vals/x_vals`), fp32 SiLU (`out_val/(1+expf(-out_val))`), single cast on
store. Parity vs `F.conv1d(groups=dim)` at bf16 `atol≈2e-2/rtol≈1e-2`. Causal left-pad + in-place state
shift are the correctness-critical parts (off-by-one in `update_idx` → plausible-but-wrong window) — see
[../numerics.md](../numerics.md).

## Integration (rebind seam)
`aiter.ops.causal_conv1d.causal_conv1d_update(x, conv_state, weight, bias, out, use_silu, cache_seqlens,
conv_state_indices, pad_slot_id)` — note this C++ entry takes an explicit `out` tensor and `use_silu`
bool (vs the Triton wrapper's `activation` str). Pass empty tensors (`torch.empty(0,...)`) for optional
args. `out` must be `zeros_like` when pad slots are used (skipped rows are not written). Wire at the GDN
mixer decode call site; e2e-gate on decode tok/s.

## Pitfalls & anti-patterns
- ⚠ **Width restricted to 2/3/4** (`TORCH_CHECK(width>=2 && width<=4)`); other widths abort.
- ⚠ `out = empty_like` + pad slots → garbage in padded rows (kernel early-returns on `pad_slot_id`).
- ⚠ This file is **decode-only** — there is no HIP varlen-prefill conv here; do not assume it covers
  `causal_conv1d_fn`. Prefill is Triton.
- First call JIT-compiles the module (latency spike) — warm it before timing/serving.
- `conv_state` is updated in place and unguarded across streams — don't alias it concurrently.

## How to verify
Build/warm the module, then `aiter/op_tests/test_causal_conv1d.py` for parity; multi-step decode vs a
rolling `F.conv1d` reference (catches state-index bugs). ISA spot-check (`--save-temps`): want vectorized
`global_load`, no `scratch_` spills, the tap loop unrolled, no `ds_*` (LDS) traffic.

## Alternatives / cross-links
[triton.md](triton.md) (varlen prefill + decode) · [aiter.md](aiter.md) (dispatcher) ·
[../overview.md](../overview.md) · language: [`../../../languages/hip_cpp/`](../../../languages/hip_cpp/) ·
op cross-link: [[linear_attention_gated_delta]].

## Sources
- HIP decode kernel (64-thread wave, register sliding-window, template width/circular, fp32 MAC+SiLU): `ROCm/aiter@a6bb49937:csrc/kernels/causal_conv1d_update.cu`.
- Python entry + docstring (out tensor, use_silu, zeros_like for pad, width 2/3/4, MI308 note): `ROCm/aiter@a6bb49937:aiter/ops/causal_conv1d.py`.
- C++ header / pybind: `ROCm/aiter@a6bb49937:csrc/include/causal_conv1d.h`, `csrc/pybind/causal_conv1d_update_pybind.cu`.
