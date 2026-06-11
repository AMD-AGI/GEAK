---
title: cumsum_scan — numerics & parity
kind: operator_overview
operator: cumsum_scan
gens: [gfx942, gfx950]
dtypes: [fp32, bf16, fp16, int32, int64]
regimes: [both]
updated: 2026-06-08
sources:
  - https://github.com/triton-lang/triton/issues/2359
  - https://github.com/triton-lang/triton/issues/3017
  - https://triton-lang.org/main/python-api/generated/triton.language.cumsum.html
---

# cumsum_scan — numerics & parity

Scan inherits reduction's order-dependence and adds a **sharp Triton correctness trap** for
non-commutative combine functions (exactly the SSM/gated-recurrence case).

## 1. ⚠ Triton `associative_scan` operand-order bug (non-commutative ops)
Triton issue #2359: `tl.associative_scan` gives **incorrect results for non-commutative `combine_fn`**.
- For **seq ≤ 64**: the correct result can be recovered by **swapping the operand order** in your combine.
- For **seq ≥ 128**: **no workaround** — the first 64 outputs are correct (with the swap), the second 64
  are always wrong. The cause: the merge step uses the right order but the small-sequence accumulate uses
  `combine_fn(b, a)` instead of `combine_fn(a, b)`.
This directly affects **SSM/gated-delta/EMA pair-scans** (the pair operator is non-commutative). Mitigation:
keep the per-program scan length ≤ 64 with the operand swap, or do the recurrence with a hand-written HIP
scan ([backends/hip.md](backends/hip.md)) where you control order. **Always test seq=64 AND seq≥128**
against a torch reference.

## 2. ⚠ Don't mix `tl.sum` and `tl.cumsum` in one kernel
Issue #3017: a kernel using both `tl.sum` and `tl.cumsum` produced results that diverged from torch.
Split them into separate kernels or validate carefully.

## 3. fp accumulation order
Scan combines along the axis in a **tree**, a different order than torch's sequential `cumsum`, so bf16/fp16
outputs differ in the LSB. **Accumulate in fp32** (`tl.cumsum` auto-promotes bf16→fp32; small ints upcast
to avoid overflow). For a long cumsum the running value can grow large — fp32 is needed even for fp16 I/O
to avoid catastrophic rounding late in the sequence.

## 4. Inclusive vs exclusive & reverse
Off-by-one between inclusive (`out[i]` includes `x[i]`) and exclusive (excludes it) is the most common
*logic* bug, especially for **MoE offsets** (you usually want **exclusive** scan = start offset of each
expert's bucket). `reverse=True` scans from the right — verify the direction matches the consumer.

## 5. Integer scans (MoE histograms)
MoE token-count cumsum is **integer** and **exact** (no fp issue) — but watch overflow: a cumsum over many
experts × large batch can exceed int32; `tl.cumsum` upcasts <32-bit but int32→int64 is on you. Exactness
means bitwise parity vs torch is expected here (unlike fp).

## Parity gate
- fp scan: fp32 atol vs `torch.cumsum` / a reference tree; test seq=64 and seq≥128.
- non-commutative combine (SSM): explicit reference recurrence, both seq regimes — the #2359 bug is silent.
- integer scan (MoE): bitwise parity vs torch; check exclusive-vs-inclusive and overflow.

## Sources
- associative_scan wrong for non-commutative ops; seq≤64 swap workaround, seq≥128 unfixable: https://github.com/triton-lang/triton/issues/2359
- tl.sum + tl.cumsum same-kernel wrong results: https://github.com/triton-lang/triton/issues/3017
- tl.cumsum dtype upcast (<32-bit, bf16→fp32), reverse: https://triton-lang.org/main/python-api/generated/triton.language.cumsum.html
