# Gluon — RDNA WMMA (downgrade target)

**Downgrade** alongside gfx942 — not the primary CDNA optimization line. Triton exposes
`gl.amd.rdna3.wmma` and `gl.amd.rdna4.wmma`; use when the contract targets client GPUs
(gfx11* / gfx120*).

ISA reference: `../hardware/rdna-fork.md`.

## API

```python
# gfx11* — v16-operand ABI; f16/bf16/i8/i4; NO fp8
acc = gl.amd.rdna3.wmma(a, b, acc, ...)

# gfx120* — v8-operand ABI; + fp8/bf8
acc = gl.amd.rdna4.wmma(a, b, acc, ...)
```

Lowering: `_wmma(1,...)` vs `_wmma(2,...)` in Triton AMD backend.

## Layout — `AMDWMMALayout` (v8 vs v16 ABI)

`AMDWMMALayout` — **not** `AMDMFMALayout`. Recover via TTGIR transcription; the
per-lane operand/result register map differs by WMMA version:

| ABI | Arch | C/D fragment | Lane map | Notes |
| --- | --- | --- | --- | --- |
| **v16-operand** | RDNA3 (gfx11*) | ~8 VGPRs/lane (16×16) | `_w32` duplicates operands across the 2 halves of the wave | `gl.amd.rdna3.wmma`, `_wmma(1,...)` |
| **v8-operand** | RDNA4 (gfx120*) | denser (~half the operand VGPRs of v16) | v8 lane map — **different**, do not reuse the v16 store map | `gl.amd.rdna4.wmma`, `_wmma(2,...)` |

Practical consequence: an RDNA3 store/convert layout **will not** be correct on RDNA4 —
transcribe the `#amd_wmma` layout from the actual gfx1201 TTGIR, do not port the gfx11
map. Exact v8 basis vectors: recover from TTGIR / verify against the Triton
`amd.rdna4.rst` API doc (`unknown-needs-probe` until confirmed on the build).

### Constructor + TTGIR field mapping

`AMDWMMALayout` exists in `gluon.language.amd` — an `amd_wmma` layout **is**
transcribable (`ttgir_to_gluon.py` emits it — gluon pack). Two generations of the
signature ship, and TTGIR moved with it, so read the mapping off the spelling **in your
dump** rather than assuming:

```python
# current: bases given directly (TTGIR prints ctaLayout = {...})
gl.amd.AMDWMMALayout(version, transposed, warp_bases,
                     reg_bases=None, instr_shape=None, cga_layout=[], rank=None)
# older:   flat warpsPerCTA (TTGIR prints warpsPerCTA = [...])
gl.amd.AMDWMMALayout(version, transposed, warps_per_cta,
                     instr_shape=None, tiles_per_warp=None, cga_layout=[])
```

| TTGIR attribute | Gluon argument | Note |
| --- | --- | --- |
| `version` | `version` | 1 = RDNA3 (gfx11*), 2 = RDNA4 (gfx120*), 3 = gfx1250 |
| `isTranspose` | `transposed` | **no `d`** — MFMA spells it `isTransposed`; the MFMA name reads as absent on a WMMA layout |
| `ctaLayout = {warp = [...]}` | `warp_bases` | GF(2) bases, not a warps-per-CTA vector |
| `ctaLayout = {register = [...]}` | `reg_bases` | absent when the layout has no register repetition |
| `warpsPerCTA = [...]` (older dumps) | `warps_per_cta` | the pre-`ctaLayout` spelling; pick the matching signature |
| `instrShape` (usually absent) | `instr_shape` | v1/v2 have the single 16×16×16 intrinsic → default; v3 has several → read it off the ISA, do not guess |
| `tilesPerWarp` (older dumps) | `tiles_per_warp` | older signature only; defaults to all-ones, so a non-unit value must be carried across |
| — | `rank` | defaults to 2; pass it when the bases are not rank-2 |

`AMDMFMALayout` takes `warps_per_cta` and `instr_shape` and does **not** accept bases —
the two constructors are not interchangeable.

### The consequence people meet: `num_warps` stops being readable from inside the body

This is the part of "not interchangeable" that changes code shape rather than spelling, and it is
worth knowing before you price a port either way.

**On the MFMA side the warp grid is a value.** `warps_per_cta=[1, NUM_WARPS]` puts the wave count
*inside* a list of fixed length, so `gl.num_warps()` — a core `gl` builtin, available on every
version — is enough: the body reads it and builds the layout, one line, no branch.

**On the WMMA side the warp grid is the length and contents of a basis list.** `log2(num_warps)`
bases, each a vector. No expression over a constexpr can produce a list whose *length* varies, and
a traced body cannot build a Python list, so the bases have to be fixed before the layout is
constructed. That is a real constraint and not a style preference; it has three known answers:

| | shape | what it costs |
| --- | --- | --- |
| **literal ladder in the body** | `if num_warps == 1: bases = []` / `elif == 2: [[1,0]]` / `elif == 4: [[1,0],[2,0]]` / `else: [[1,0],[2,0],[4,0]]` | most direct, and it pins `num_warps` into the **kernel signature** — the value now has to be known at trace time, where the MFMA side could read it in the body. One surveyed kernel pays ~25 lines for what is one line on the MFMA side. |
| **compute on the host and pass the bases in** | `for i in range(log2(nw // 2)): bases.append((1 << i, 0))`, result handed in as a constexpr | moves the constraint out of the kernel, so the body stays polymorphic. Usually the best of the three. |
| **`@constexpr_function` lookup** | a dict keyed by the tile dimension, evaluated at trace time | an unsupported combination raises `KeyError` instead of silently taking a default branch (`../tile-programming/layout-recipes.md ### Put a gate on the factory`) |

**So when you cost a port between the two matrix layouts, do not cost it as an API rename.** The
basis list is restated in every layout in the file — blocked, linear and dot-operand — and they all
move together with the warp mapping; that is the same coupling that makes `warps_per_cta` a
multi-round rewrite rather than a knob on the MFMA side (`../pitfalls/negative-patterns.md`). Grep the
source for `warp_bases` first; the count is the size of the change.

## Pipeline

No `num_stages`; hand LDS ping-pong (2-stage typical on gfx1151-class). No async
cp/TDM on RDNA4 ISA — TDM is gfx1250 (CDNA5 / MI450) only, a separate data-center
fork, not RDNA4.

## Low precision (FP8 / FP4 handling)

RDNA4 has **FP8/BF8 WMMA** (`V_WMMA_F32_16X16X16_FP8_*`, e4m3 / e5m2 -> fp32 acc) but
**no** `mfma_scaled` / OCP MXFP4 hardware path and **no** block-scale unit:

- **FP8 scale:** there is no fused block-scale in the WMMA op — any per-block scale
  must be applied **outside** the matrix instruction (dequant to bf16 then WMMA, or fold
  the scale into the epilogue / a VALU side path). Do **not** copy the CDNA
  `mfma_scaled` (E8M0 block scale) recipe here — it is a scoped ceiling on RDNA.
- **FP4:** no OCP MXFP4 tensor path — use **IU4 WMMA** (integer 4-bit) or dequant+WMMA.
- Exact Gluon FP8 WMMA operand/format arguments: verify against `amd.rdna4.rst` +
  a compile-verified probe (`unknown-needs-probe`).

## Structured sparsity (SWMMAC)

RDNA4 ISA has **SWMMAC 4:2** structured-sparse WMMA (2× dense throughput, e.g. the
sparse columns in `../hardware/amd-rdna4-skus.md`). The sparse operand needs a **sparsity index**
(metadata selecting the 2 non-zero of every 4 elements) alongside the compressed
operand. The **Gluon API path for SWMMAC is `unknown-needs-probe`** — the ISA supports
it, but exposure via `gl.amd.rdna4.*` + the index-operand layout must be probed on the
build before planning a sparse kernel. Do not assume it from the dense WMMA path.

## Escalation rule

Do **not** use a CDNA MFMA Gluon anchor as the starting point for RDNA — warm-start
from [Triton Gluon RDNA3/RDNA4 API docs](https://github.com/triton-lang/triton/tree/main/docs/gluon/api) (`amd.rdna3.rst`, `amd.rdna4.rst`) or [ROCm/FlyDSL RDNA kernels](https://github.com/ROCm/FlyDSL/tree/main/kernels).

## Smoke

```bash
# minimal @gluon.jit with rdna4.wmma on gfx1201 — verify wave32, WMMA in ISA dump
```
