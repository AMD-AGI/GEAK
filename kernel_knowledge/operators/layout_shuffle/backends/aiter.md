---
title: layout_shuffle on aiter — SOTA card
kind: sota_card
operator: layout_shuffle
backend: aiter
gens: [gfx942, gfx950]
dtypes: [bf16, fp8_e4m3_fnuz, fp4_e2m1, int8]
regimes: [prefill, decode, both]
status: sota
updated: 2026-06-08
sources:
  - ROCm/aiter@a6bb499375849eec45d68c5ccaebc8865fd422c0:aiter/ops/shuffle.py
  - ROCm/aiter@a6bb499375849eec45d68c5ccaebc8865fd422c0:aiter/tuned_gemm.py
  - ROCm/aiter@a6bb499375849eec45d68c5ccaebc8865fd422c0:aiter/configs/
---

# layout_shuffle × aiter

## TL;DR
`aiter.ops.shuffle.shuffle_weight` is **the** weight pre-shuffle on AMD: it permutes a weight into the
MFMA-friendly layout, sets `is_shuffled=True`, and thereby flips the **`bpreshuffle`** field of the dense-GEMM
9-tuple dispatch key so the GEMM/MoE dispatches to the fast **bpreshuffle** asm/CK/FlyDSL/FP4 kernels. Tuned
bpreshuffle CSVs ship for a8w8 / a8w8-blockscale / a4w4. Verified on-box at `ROCm/aiter@a6bb49937`.

## SOTA implementation(s)
| impl | source | gens/dtypes | measured perf | when best |
|---|---|---|---|---|
| `shuffle_weight(W, layout=(IN,IK))` → bpreshuffle GEMM | `aiter/ops/shuffle.py` + `aiter/tuned_gemm.py` | gfx942/950, bf16/fp8/int8 | one-time; **unlocks** the bpreshuffle GEMM kernels (the GEMM speedup is the Amdahl lever — [[operators/dense_gemm/overview.md]]); shuffle itself not separately timed | any bpreshuffle GEMM path |
| `shuffle_weight_a16w4` + `shuffle_scale_a16w4` → FP4 MoE | `aiter/ops/shuffle.py` | gfx950, fp4 | feeds `f4gemm_*_BpreShuffle_*` (part of up-to-3× fused MoE, AMD-reported) | FP4 MoE expert weights |

## Config space / knobs
- **layout**: `(16,16)` default; `(32,16)` a8w8 asm; FlyDSL `(16·pack_n,16)`; FP4 `NLane`/`gate_up`.
- **`use_int4=True`** for 4-bit packing (`Kel=32`).
- **bpreshuffle CSVs**: `AITER_CONFIG_GEMM_A8W8_BPRESHUFFLE`, `..._BLOCKSCALE_BPRESHUFFLE`,
  `AITER_CONFIG_GEMM_A4W4` (`:`-mergeable) — must contain `bpreshuffle=True` rows.
- Tune via gradlib with the bpreshuffle libtype; gate `err_ratio<0.05`.

## Numerics / parity
value-preserving permutation (exact); weight **and** its FP4/FP8 block scale must be shuffled together; the
only failure is layout-vs-kernel mismatch (gross, caught by GEMM `allclose`). See
[[operators/layout_shuffle/numerics.md]].

## Integration (rebind seam)
Shuffle at model load (the quant/loader path), then the runtime `gemm_a16w16`/`gemm_a8w8`/`gemm_a4w4` /
`fused_moe` reads `B.is_shuffled` and dispatches bpreshuffle automatically. Deploy tuned bpreshuffle rows by
env. Verify the row hits with `AITER_LOG_TUNED_CONFIG=1`.

## Pitfalls & anti-patterns
- ⚠ **bpreshuffle key mismatch = 0 engagement**: a tuned row with `bpreshuffle=False` won't match a shuffled
  call (and vice versa) — same trap class as `bias`/`cu_num` ([[backends/aiter/tuned_gemm.md]]).
- ⚠ Forgetting to shuffle the FP4 **scale** alongside the weight → wrong dequant.
- ⚠ Wrong `layout=` for the consuming kernel → garbage GEMM output.
- ⚠ Dims must divide (`N%IN==0`, `K%BK==0`, FP4 `real_k≥256`) — pad otherwise.

## How to verify
`AITER_LOG_TUNED_CONFIG=1` → bpreshuffle row hit; GEMM output `allclose` vs unshuffled-weight GEMM; rocprofv3
→ the bpreshuffle asm/CK/FP4 kernel fires.

## Alternatives / cross-links
[backends/hip.md](hip.md) · [backends/triton.md](triton.md) · [[operators/dense_gemm/backends/aiter.md]] ·
[[backends/aiter/tuned_gemm.md]] · [[backends/aiter/configs_db.md]] · [[backends/aiter/fmoe.md]].

## Sources
- On-box: `ROCm/aiter@a6bb499375849eec45d68c5ccaebc8865fd422c0`: `aiter/ops/shuffle.py` (shuffle math, layouts, is_shuffled), `aiter/tuned_gemm.py` (bpreshuffle key), `aiter/configs/` (bpreshuffle CSVs, FP4 BpreShuffle kernel names).
