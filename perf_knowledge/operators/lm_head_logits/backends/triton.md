---
title: lm_head_logits on triton — SOTA card
kind: sota_card
operator: lm_head_logits
backend: triton
gens: [gfx942, gfx950, gfx1151]
dtypes: [bf16, fp16, fp32]
regimes: [decode, prefill, both]
status: competitive
updated: 2026-06-08
sources:
  - https://github.com/triton-lang/triton/blob/main/third_party/amd/backend/compiler.py
  - https://rocm.docs.amd.com/en/latest/how-to/rocm-for-ai/inference-optimization/workload.html
  - ROCm/aiter@a6bb499375849eec45d68c5ccaebc8865fd422c0:aiter/tuned_gemm.py
---

# lm_head_logits × triton

## TL;DR
Triton is **competitive, not default**, for the head GEMM: on a plain `(M=batch, N=V, K=d)` projection it
loses to tuned hipBLASLt/AITER. Where Triton earns its place is **fusion** — a single kernel that does the
skinny split-K GEMM **and** the soft_cap/scale/bias epilogue (and optionally an argmax), avoiding the extra
`[M,V]` pass at V=128k–256k. It is also the fallback aiter dispatches to (`aiter.ops.triton.gemm`) when no
tuned asm/CK path exists for the head shape. Author Triton here to fuse, not to beat the library at bare
GEMM.

## SOTA implementation(s)
| impl | source | gens/dtypes | measured perf | when best |
|---|---|---|---|---|
| Triton skinny split-K GEMM + fused epilogue | author (patterns: [[triton_amd]] patterns.md) | gfx942/950, bf16/fp16 | bare GEMM ~loses to hipBLASLt; **wins by fusing soft_cap/scale into one pass** at large V | fused-epilogue head, or a shape with no tuned asm path |
| `aiter.ops.triton.gemm.basic.gemm_a16w16` (aiter's Triton libtype) | `ROCm/aiter@a6bb49937:aiter/ops/triton/` | gfx942/950, bf16/fp16 | portable fallback when DB has no asm/CK winner | aiter dispatches it for uncovered head shapes |

## Config space / knobs
Skinny GEMM (`M` tiny, `N=V` huge, `K=d`) — see [[triton_amd]] knobs.md:
- `BLOCK_M`∈{16,32,64} (small — M is batch), `BLOCK_N`∈{128,256}, `BLOCK_K`∈{32,64,128}.
- **`SPLIT_K`∈{2,4,8,16}** — the key knob: with M tiny, split the K=d reduction so M·(V/BLOCK_N)·SPLIT_K
  reaches ≥1024 programs across 304 CUs (the `[V,d]` weight read is the bottleneck → maximize parallel
  loads). Costs a C zero-init + atomics.
- `matrix_instr_nonkdim=16`, `num_warps=4` (wave64; not 8), `num_stages=2` (single GEMM); `kpack=2` for
  bf16 `BLOCK_K≥64` on gfx942. `OPTIMIZE_EPILOGUE=1`.
- `GROUP_SIZE_M`=8 matters little (M is tiny); the grid is N-dominated.

## Numerics / parity
fp32 accumulate, **fp32 logits out**. Fuse soft_cap (`tanh`) on the fp32 accumulator before any downcast.
Argmax-fusion tie-break = lowest index (match reference). Triton vs hipBLASLt reduction order differs →
re-check greedy/temp=0 parity (see [../numerics.md](../numerics.md)).

## Integration (rebind seam)
Two ways in: (1) aiter's Triton libtype is selected automatically per-shape from its DB (no user action);
(2) hand-author a fused head GEMM and register it as the model's head op (TorchInductor or a custom op).
Inductor `max-autotune` can also emit the head `addmm` as Triton with AMD knobs wired in. Verify: rocprofv3
shows a Triton-named kernel (Python symbol) for the `N=V` GEMM rather than `Cijk_*`.

## Pitfalls & anti-patterns
- Expecting Triton to beat tuned hipBLASLt on the **bare** head GEMM — it won't; the value is fusion.
- `num_warps=8` (NVIDIA habit) → wave64 VGPR spill ([[triton_amd]] pitfalls).
- No `SPLIT_K` at tiny M → grid underfills 304 CUs, weight read serializes → slow.
- fp16 logits out → range clip; keep fp32.

## How to verify
`TRITON_PRINT_AUTOTUNING=1` for the winning config; isolated bench vs hipBLASLt at `(M,V,d)`; greedy parity
after swap; confirm a fused kernel (one `[M,V]` pass, not GEMM-then-epilogue).

## Alternatives / cross-links
[aiter.md](aiter.md) (live GEMM) · [vllm_kernels.md](vllm_kernels.md) · [hip.md](hip.md) ·
[../overview.md](../overview.md) · [[triton_amd]] · [[skinny_gemv_decode]] · [[gemm_epilogue_fused]].

## Sources
- AMD Triton backend knobs / split-K / Inductor codegen: https://github.com/triton-lang/triton/blob/main/third_party/amd/backend/compiler.py
- Skinny/split-K decode GEMM, ≥1024 grid: https://rocm.docs.amd.com/en/latest/how-to/rocm-for-ai/inference-optimization/workload.html
- aiter's Triton GEMM libtype (fallback dispatch): `ROCm/aiter@a6bb499375849eec45d68c5ccaebc8865fd422c0:aiter/tuned_gemm.py`.

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
