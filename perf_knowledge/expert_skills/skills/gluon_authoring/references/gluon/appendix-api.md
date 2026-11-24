# Appendix: single-point Gluon APIs (gfx950 / gfx942)

APIs too small for a mechanism section, and spellings that **do not exist**. Source-proven against
3.6.0 / 3.7.0 / 3.7.1 / 3.8.0 (not timed). Where a mechanism page disagrees, the mechanism page wins.

## Spellings that do not exist on the CDNA Gluon path

| Reached for | Status |
| --- | --- |
| `pipelined_range` | **Does not exist** upstream in `gl` or `tl` on 3.6.0–3.8.0 (3.8.0 re-checked by `hasattr`). `tl.range(..., num_stages=)` is no substitute: no pass reads `num_stages` on the Gluon path (`pipeline/loop-knobs-and-targets.md ## Roll the loop (cut i-cache pressure)`). Depth = buffers you allocate, or plain's pipeliner re-injected (`pipeline/reinjection.md`). |
| `volatile=True` on `gl.load` | **Reachable** (`gl.load` is `builtin(tl_core.load)`; production passes it). On AMD (Triton 3.8, gfx942/gfx950) it sets **no cache bits**, and a poll of one shared flag address written with it (or as a plain load) is hoisted out of the spin loop even with `disable_licm=True`, so the loop never re-reads the flag. Not a cache-bypass or polling control: poll with `cache_modifier=".cv"` (emits `sc0 sc1` and stays in the loop) or an acquire atomic (`collective.md`, `memory-reference.md`). |
| `volatile=` on `gl.store`, cdna3/cdna4 buffer ops, `global_load_to_shared` / `buffer_load_to_shared` | **Not a parameter.** |
| `input_precision=` on `gl.amd.cdna3.mfma` / `cdna4.mfma` | **Not a parameter** — signature is `mfma(a, b, acc)`. The dot is tagged from the process-wide **`TRITON_F32_DEFAULT`** env knob (else the `default_dot_input_precision` compile option), but on 3.8.0 the Gluon pipeline does not run the `f32_dot_tc` pass, so neither changes the emitted `v_mfma_f32_*_f32` (compile-verified gfx950). fp32 decomposition in Gluon is written out by hand (`../tile-programming/low-precision.md ### One-sided decomposition: when only A is f32`). `gl.dot_fma` passes `input_precision=None`. No `allow_tf32` / `bf16x3` spelling. (It *is* a parameter of `gl.nvidia.ampere.mma_v2`.) |
| `fence_async_shared` | **NVIDIA-only** (`gl.nvidia.hopper`, re-exported by `blackwell`); no `gl.amd.*` counterpart. The CDNA4 async surface is `global_load_to_shared`, `buffer_load_to_shared`, `commit_group`, `wait_group`, `load_shared_relaxed` — none is a fence; fill-vs-read ordering: `pipeline/async-ordering.md ## Ordering the fill against the read — class S`. |
| pointer-form `smem.atomic_add` | Does not exist; use `atomic_scatter_add` (3.8.0; `smem-lds-reference.md ## Atomic RMW in LDS — the atomic_scatter_* family`). |
| `gl.range`, `gl.sort`, `gl.topk`, `gl.cumsum`, `gl.argmax` / `argmin`, `gl.flip`, `gl.interleave` | **Absent from `gl`** (3.8.0); loops use Python `range` / `gl.static_range` / `tl.range` (where `loop_unroll_factor=` lives). |

Names that **do** exist in `gl` on 3.8.0 and are easy to assume missing: `map_elementwise`, `reduce_or`,
`xor_sum`, `ravel`, `zeros_like`, `full_like`, `num_programs`, `associative_scan`, `histogram`,
`join` / `split`, `clamp`, `fma`, `umulhi`, `div_rn` / `sqrt_rn`, `inline_asm_elementwise`, `dot_fma`
(checked with `hasattr(gl, name)`).

## The compile-time oracle worth knowing about

**`gl.bank_conflicts(distr_ty, shared_ty) -> int` exists on all four versions but SIGABRTs the
Python process on the 3.8.0 ROCm wheel for every layout pair** (`inDim not found in layout` for
Blocked / linear / dot-operand vs `SwizzledSharedLayout`; `unknown layout` for
`PaddedSharedLayout`); inside `gl.static_assert` it aborts the compile the same way (re-checked 3.8.0,
gfx950 + gfx942). The working checks are the ISA `ds_read` census and `SQ_LDS_BANK_CONFLICT`
(`layout-reference.md ### Checking the choice at compile time`).
Were it working, it would return excess `ld.shared`/`st.shared` accesses per wavefront for that
pair: a static count, blind to transposed reads and global memory.

## Layout-carrying vs bare re-export — the distinction that bites

| API | All four versions |
| --- | --- |
| `gl.histogram(input, num_bins, mask=None, layout=None)` | **`layout=` is mandatory despite the `None` default** — omitting it raises *"histogram requires a destination layout"* (re-checked 3.8.0). Input 1D integer. |
| `gl.associative_scan` | Bare re-export: **no** `layout=` kwarg. |
| `gl.assume`, `gl.max_constancy`, `gl.max_contiguous`, `gl.multiple_of` | Bare re-exports, identical to `tl`. |

## Math

Gluon's math set (`div_rn`, `umulhi`, `exp`, `exp2`, `fma`, `log`, `log2`, `cos`, `rsqrt`, `sin`,
`sqrt`, `sqrt_rn`, `abs`, `fdiv`, `erf`, `floor`, `ceil`) is identical on all four versions.
Everything else goes through `gl.extra.libdevice` (bare re-export of `tl.extra.libdevice`, no layout
handling). `enable_fp_fusion` is a module-wide compile option, not a `gl` API
(`../workloads/reduction-elementwise.md ## Stage 3 — dequant + residual + norm`). Descriptor
`._reinterpret` / `.gather` / `.scatter`: `smem-lds-reference.md ## Indexed access to LDS — .gather / .scatter`.

## A naming trap: `gl.amd.cdna5`

**`gl.amd.cdna5` = gfx1250, not a gfx950 successor.** It is a star re-export of `gl.amd.gfx1250`
(present on some 3.8.0 trees, absent from the installed ROCm wheel). The CDNA namespaces for
gfx942 / gfx950 are `cdna3` / `cdna4`. Namespace is not architecture support
(`../hardware/capability-matrix.md`).

## What upstream cannot tell you here

- **Upstream ships no gfx950 / gfx942 Gluon example kernel.** `third_party/amd/python/examples/gluon/`
  is AMD-authored but **every file is `*_gfx1250.py`** (wave32, `AMDWMMALayout`, TDM, `mbarrier`);
  `AMDMFMALayout`, `buffer_load_to_shared`, `mfma_scaled`, `gl.amd.slice`, `inline_asm_elementwise`
  appear zero times. `python/tutorials/gluon/` and `python/examples/gluon/` are NVIDIA.
- CDNA3/CDNA4 Gluon is exercised in `python/test/gluon/test_frontend.py` (`HIP_TARGET_CDNA3` /
  `HIP_TARGET_CDNA4`: MFMA layouts, `mfma` / `mfma_scaled`, `scaled_upcast`, `PaddedSharedLayout`,
  `buffer_atomic_rmw`, async copy, `warp_pipeline_stage`) — read for spelling, not for kernel
  composition (bodies are degenerate; they assert IR).
