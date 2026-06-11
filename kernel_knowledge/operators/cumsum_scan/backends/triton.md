---
title: cumsum_scan on Triton — SOTA card
kind: sota_card
operator: cumsum_scan
backend: triton
gens: [gfx942, gfx950]
dtypes: [fp32, bf16, fp16, int32]
regimes: [both]
status: sota
updated: 2026-06-08
sources:
  - https://triton-lang.org/main/python-api/generated/triton.language.associative_scan.html
  - https://triton-lang.org/main/python-api/generated/triton.language.cumsum.html
  - https://github.com/triton-lang/triton/issues/2359
  - https://github.com/proger/accelerated-scan
---

# cumsum_scan × triton

## TL;DR
Triton has **first-class scan primitives**: `tl.cumsum(x, axis, reverse)` for plain prefix-sum and
`tl.associative_scan(x, axis, combine_fn, reverse)` for general recurrences (SSM/gated-delta/EMA). They
lower to the wave-scan + LDS automatically. This is the SOTA authoring path for MoE-offset cumsum and
linear-attention scans — **with one sharp caveat**: `associative_scan` is **buggy for non-commutative
combines at seq ≥ 128** (issue #2359). Plain `tl.cumsum` is fine.

## SOTA implementation(s)
| impl | source | gens/dtypes | notes | when best |
|---|---|---|---|---|
| `tl.cumsum` row scan | this card | gfx942/950, fp32 acc | plain prefix-sum, safe | MoE offsets, CDFs, running sums |
| `tl.associative_scan` pair-scan | `accelerated-scan` (Triton path) | gfx942/950 | gated recurrence; **seq ≤ 64** to dodge #2359 | linear-attn / SSM, short chunks |
| chunked 3-stage scan | driver + kernel (tuning.md) | gfx942/950 | long axis | sequence-length scans |

```python
@triton.jit
def plus(a, b): return a + b

@triton.jit
def row_cumsum(x_ptr, o_ptr, sr, n, BLOCK: tl.constexpr):
    r = tl.program_id(0); cols = tl.arange(0, BLOCK)
    x = tl.load(x_ptr + r*sr + cols, mask=cols < n, other=0.0).to(tl.float32)
    y = tl.cumsum(x, 0)                              # wave-scan + LDS; or associative_scan(x,0,plus)
    tl.store(o_ptr + r*sr + cols, y, mask=cols < n)
# grid = (rows,);  BLOCK = next_pow2(n)
```

## Config space / knobs
- `BLOCK = next_pow2(chunk)`; one program per row for short-axis many-row (MoE).
- `num_warps`: 2/4 (how many waves combine via LDS); `num_stages=1`.
- `reverse=True` for right-to-left; `tl.cumsum(dtype=...)` to force the accumulate dtype.
- For non-commutative combines: **keep per-program seq ≤ 64** + operand-swap (issue #2359 workaround), or
  use [hip.md](hip.md).
- Long axis: 3-stage stitch in the driver (block-scan → carry-reduce → carry-add).

## Numerics / parity
fp32 accumulate (auto for bf16; ints upcast <32-bit). ⚠ `associative_scan` non-commutative bug at
seq ≥ 128; ⚠ don't mix `tl.sum`+`tl.cumsum` in one kernel (#3017). Exclusive-vs-inclusive off-by-one for
MoE offsets. See [../numerics.md](../numerics.md).

## Integration (rebind seam)
Inductor lowers `torch.cumsum` to a Triton scan; for MoE/SSM the scan is inside a hand-written fused kernel
registered as a torch custom op. `accelerated-scan` is the reference Triton library for first-order scans.

## Pitfalls & anti-patterns
- **#2359**: silent wrong results for SSM pair-scans at seq ≥ 128 — test both regimes.
- `BLOCK < axis` for a long scan → need chunking (a single program can't scan an arbitrarily long axis).
- Inclusive vs exclusive mismatch for MoE offsets.
- Mixing reduce + scan in one kernel (#3017).

## How to verify
fp32 atol vs `torch.cumsum`; **explicitly test seq=64 and seq=128** for any custom combine; for MoE,
bitwise-int parity + exclusive-offset check.

## Alternatives / cross-links
[hip.md](hip.md) (order-controlled scan, no #2359) · [../tuning.md](../tuning.md) · [../fusion.md](../fusion.md)
· [`../../../languages/triton_amd/patterns.md`](../../../languages/triton_amd/patterns.md).

## Sources
- `tl.associative_scan` (combine_fn, reverse): https://triton-lang.org/main/python-api/generated/triton.language.associative_scan.html
- `tl.cumsum` (dtype upcast, reverse): https://triton-lang.org/main/python-api/generated/triton.language.cumsum.html
- non-commutative bug seq≥128: https://github.com/triton-lang/triton/issues/2359
- accelerated-scan Triton first-order scan: https://github.com/proger/accelerated-scan
