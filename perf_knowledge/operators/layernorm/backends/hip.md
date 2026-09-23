---
title: layernorm on hip — SOTA card
kind: sota_card
operator: layernorm
backend: hip
gens: [gfx908, gfx90a, gfx942, gfx950, gfx1151]
dtypes: [bf16, fp16, fp32]
regimes: [both]
status: sota
updated: 2026-06-08
sources:
  - https://github.com/vllm-project/vllm/blob/main/csrc/layernorm_kernels.cu
  - https://rocm.docs.amd.com/projects/HIP/en/latest/reference/kernel_language.html
  - https://gpuopen.com/learn/amd-lab-notes/amd-lab-notes-matrix-cores-readme/
---

# layernorm × hip

## TL;DR
Hand-written HIP is the reference/ceiling for LayerNorm. vLLM's `csrc/layernorm_kernels.cu` carries the
LayerNorm path alongside RMSNorm; the kernel is two **wave64 block-reductions** (μ then σ²) over a
128-bit-vectorized row, fp32 accumulate, `γ,β` fp32. Reach for HIP for a fusion the library lacks or to own
the ISA.

## SOTA implementation(s)
| impl | source | gens/dtypes | measured perf | when best |
|---|---|---|---|---|
| vLLM HIP LayerNorm kernel | `vllm/csrc/layernorm_kernels.cu` | gfx942/950, bf16/fp16 | bandwidth-bound; vectorized I/O (PR #22602 family) | editable HIP / non-AITER |
| aiter asm/CU `module_norm` | `/sgl-workspace/aiter/aiter/ops/norm.py` | gfx942/950 | floor | aiter path ([aiter.md](aiter.md)) |
| AMD lab-notes block-reduce template | https://gpuopen.com/learn/amd-lab-notes/ | all | reference | from-scratch |

## Config space / knobs
- Block `min(next_pow2(N),1024)`, ×64; one block/row (prefill) or persistent grid-stride (decode).
- Two reductions: `__shfl_down` wave64 for μ, then for σ² over `(x−μ)²` (two-pass; or fuse a Welford
  running pair). `cub::BlockReduce` is fine but pin CCCL (CUDA13/CCCL3 broke `cub::Sum`, vLLM #24464).
- Vector I/O: `float4`/`__half2`, `__restrict__`, 16-B alignment (N%8). `γ,β` cached in LDS, reused.
- `__launch_bounds__(block, 4)` (VGPR-light). `hipcc --offload-arch=gfx942 -O3`.

## Numerics / parity
fp32 μ/σ²; **two-pass to avoid `Σx²−μ²` cancellation**; γ,β fp32-promote (vLLM #42325 class); biased
variance (/N). See [../numerics.md](../numerics.md).

## Integration (rebind seam)
- vLLM: edit `csrc/layernorm_kernels.cu`, rebuild; op in `torch_bindings.cpp` (`_C::layer_norm`).
- Standalone: `torch.utils.cpp_extension` / `TORCH_LIBRARY`.

## Pitfalls & anti-patterns
- ⚠ One-pass `Σx²−μ²` → negative σ² in bf16. Two-pass or Welford.
- ⚠ γ,β in input dtype (#42325). Promote to fp32.
- `num_tokens==0` grid crash; unaligned N → scalar loads; `int` overflow at M·N>2³¹.

## How to verify
`-Rpass-analysis=kernel-resource-usage`; `--save-temps` grep `dwordx4`; fp64 oracle, σ²≥0; greedy parity.

## Alternatives / cross-links
[aiter.md](aiter.md) · [triton.md](triton.md) · [vllm_kernels.md](vllm_kernels.md) · [miopen.md](miopen.md) ·
[[languages/hip_cpp/patterns]] §1.

## Sources
- vLLM HIP layernorm kernels: https://github.com/vllm-project/vllm/blob/main/csrc/layernorm_kernels.cu.
- wave64 block reduce: https://rocm.docs.amd.com/projects/HIP/en/latest/reference/kernel_language.html, https://gpuopen.com/learn/amd-lab-notes/amd-lab-notes-matrix-cores-readme/.
- γ/β regression: https://github.com/vllm-project/vllm/issues/42325.

## On gfx1151 (RDNA3.5, Strix Halo)
`gfx1151` appears in `gens:` because this backend's **source is portable** to RDNA — it
compiles/JITs there with no vendor asset table. It is **not** a claim that anything on this
card was measured on RDNA: every perf number, ranking and tuning recipe above is CDNA
(gfx90a/942/950) evidence. Three differences bite before any of it transfers — WMMA not MFMA,
wave32 not wave64, and 40 CU behind a 32 MB MALL on ~229 GB/s shared LPDDR5X rather than HBM —
so tile shapes, occupancy targets and the roofline ceiling all move. Any `fp8_*` entry in
`dtypes:` above is CDNA-only: gfx1151 has **no fp8 matrix instruction and no block-scaled
FP4/FP6**, so an fp8 candidate there runs EMULATED — it passes correctness and loses
performance silently. See [`../../../hardware/rdna35_gfx1151/`](../../../hardware/rdna35_gfx1151/)
and MEASURE on the box.
