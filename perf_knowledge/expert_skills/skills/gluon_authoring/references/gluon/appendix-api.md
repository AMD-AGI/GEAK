# Appendix: single-point Gluon APIs (gfx950 / gfx942)

`index.md` states that this pack is organized by mechanism and is **not** an API dictionary. That
rule stands, and this page is the deliberate exception at its edge: APIs small enough that a whole
mechanism section would be padding, plus the spellings people reach for that **do not exist**.

One line per entry: what it is, and which mechanism page owns it. Nothing here restates a mechanism
— if an entry needs more than a line, it belongs on the page it points to.

Everything below is **source-proven against 3.6.0 / 3.7.0 / 3.7.1 / 3.8.0** and unprobed on
hardware. Nothing on this page has been timed.

## Spellings that do not exist on the CDNA Gluon path

Check this section first. An appendix is usually consulted to find out whether something exists, and
the entries below are reached for often enough that their absence is the appendix's most useful
content.

| Reached for | Status |
| --- | --- |
| `pipelined_range` | **Does not exist anywhere in upstream**, on any of the four versions — not under `gl`, not under `tl`. Zero occurrences. |
| `volatile=True` on `gl.load` | **Reachable.** `gl.load` is `builtin(tl_core.load)` (`v3.8.0 .../gluon/language/_core.py:128`) and `tl.load` carries `volatile=False` in its signature, so the keyword passes through, and production uses it — including in a collective. **Its lowering semantics on AMD are a separate question — see below — so do not read this row as evidence the flag does anything.** |
| `volatile=` on `gl.store`, the cdna3/cdna4 buffer ops, `global_load_to_shared` / `buffer_load_to_shared` | **Not a parameter.** `tl.store` has no `volatile` on any version checked, and neither do the AMD buffer / direct-to-LDS entry points. The original "not reachable anywhere" reading was this store-side fact over-generalised to the whole API surface. |
| `input_precision=` on a matrix op | **Not a parameter of `gl.amd.cdna3.mfma` / `cdna4.mfma`.** The signature is `mfma(a, b, acc)`; the precision argument is supplied internally from the process-level `TRITON_F32_DEFAULT` knob. It *is* a per-call parameter on `gl.nvidia.ampere.mma_v2`, which is why it gets reached for. |
| `fence_async_shared` | **NVIDIA-only.** It is defined once in the whole Gluon tree, under `gl.nvidia.hopper`, and re-exported by `gl.nvidia.blackwell`; there is no `gl.amd.*` counterpart on any of the four versions. Reached for because the CDNA async-copy path clearly needs *something* that orders a fill against the read of it — but on CDNA that ordering is two separate mechanisms, not a fence. |

On `pipelined_range`: the mechanism people want from that name is a loop that carries a pipeline
depth, and the real spelling is plain Triton's `tl.range(..., num_stages=)`. That does not rescue
you either — `num_stages` is inert on the Gluon path on all four versions (dead in 3.8.0), for the
routing reason written up in `pipeline/loop-knobs-and-targets.md ## Roll the loop (cut i-cache
pressure)`. Depth comes from hand-authored staging — the buffer count you allocate — and only as a
diagnostic or last resort from re-injecting plain's pipeliner (`pipeline/reinjection.md`). Read that
section before concluding a loop resists pipelining.

On `input_precision`: the consequence is that **fp32 precision decomposition is not a per-call
decision on this path**. The AMD matrix ops read one process-wide knob, so you cannot run one dot at
a reduced input precision and another at full precision in the same kernel by passing an argument —
if you need that, the decomposition is yours to write out — and there is a worked account of doing
so, including the axis choice and the rounding mode it depends on, at
`../tile-programming/low-precision.md ### One-sided decomposition: when only A is f32`.
`gl.dot_fma` does not even consult the
knob: it passes `input_precision=None` outright. Do not go looking for `allow_tf32` or a `bf16x3`
spelling either; neither has a counterpart here.

On `volatile`: plain `tl.load` accepts it, which is why it gets tried — and on `gl.load` it goes
through, because `gl.load` is that same builtin. What it is **not** is a knob with a verified effect
here, and the two halves of that have to be kept apart. **Reachability is settled: the parameter
exists on `gl.load` and production passes it.** What it lowers to on CDNA is **not established** in
this pack — nobody traced `volatile=True` through the AMD backend to an instruction or cache bit, so
a sweep of it is a measurement, not a derivation. The cdna3/cdna4 buffer ops and
`global_load_to_shared` / `buffer_load_to_shared` genuinely lack the parameter, as does `gl.store`.
And even on the plain path it is not a cache-bypass control on AMD: only the NVIDIA
backend's load/store lowering consumes the flag for codegen, while the AMD backend merely propagates
it when rewriting a load into an async copy. For the cache behaviour you actually wanted, use the
cache modifiers and the buffer-op controls in `memory-reference.md`.

On `fence_async_shared`: the absence earns a row because the need it names is real on CDNA and is
met by **two** calls rather than one, so searching for a single fence finds nothing and hides half
the requirement. The whole CDNA4 async-copy surface is five entry points —
`global_load_to_shared`, `buffer_load_to_shared`, `commit_group`, `wait_group`,
`load_shared_relaxed` — and none of them is a fence. What the two calls are, which one answers
which question, and where each goes:
`pipeline/async-ordering.md ## Ordering the fill against the read — class S`.

## The compile-time oracle worth knowing about

**`gl.bank_conflicts(distr_ty, shared_ty) -> int`** — present on all four versions. Given a
distributed tensor type and a shared-memory descriptor type, it returns the number of **excess**
memory accesses per wavefront that the `ld.shared` / `st.shared` instructions between them would
take. Zero means conflict-free.

Why it earns a mention in an appendix: this pack's LDS-layout work — padding and swizzle selection
in `layout-reference.md ## BlockedLayout Constraints (wave64)` and the recipes in
`../tile-programming/layout-recipes.md` — otherwise reads a bank-conflict question as something you
settle by profiling. This turns it into a value available at compile time from the two types alone,
so a padding choice can be checked while authoring rather than after a run.

Two limits, both from what it models: it describes `ld.shared` / `st.shared` between that
distributed type and that descriptor, so it says nothing about a transposed-read path or about
global-memory access; and it is a static count, not a claim about the cost of those accesses. Treat
a non-zero result as a layout defect to fix, and a zero result as one hypothesis eliminated — not as
evidence that the kernel is fast.

## Layout-carrying vs bare re-export — the distinction that bites

Most `gl` names that look like plain builtins **are** plain builtins, re-exported unchanged. A few
take a Gluon-specific `layout=`. Guessing wrong in either direction costs a compile.

| API | On all four versions |
| --- | --- |
| `gl.histogram(input, num_bins, mask=None, layout=None)` | Takes a Gluon `layout=`. The `None` default is misleading: the destination layout is **mandatory**, and omitting it raises *"histogram requires a destination layout"*. Input must be 1D and integer. |
| `gl.associative_scan` | Bare re-export. **No** `layout=` kwarg. |
| `gl.assume`, `gl.max_constancy`, `gl.max_contiguous`, `gl.multiple_of` | Bare re-exports of the plain builtins; behaviour and signature identical to `tl`. |

The pairing of the first two rows is the one to remember, because they are adjacent in use: a
routing or bucketing step often wants both, and only one of them wants a layout. It is also why
`../workloads/moe.md` and `../workloads/linear-attention.md` both call out the scan's missing kwarg where
they use it.

## Math

`gl.div_rn` — round-to-nearest-even division — sits in Gluon's own curated math module alongside
`umulhi`, `exp`, `exp2`, `fma`, `log`, `log2`, `cos`, `rsqrt`, `sin`, `sqrt`, `sqrt_rn`, `abs`,
`fdiv`, `erf`, `floor` and `ceil`. That set of seventeen is **byte-identical across all four
versions**; there is no version gate to check on any of them.

Anything outside those seventeen goes through **`gl.extra.libdevice`**, a one-line re-export of the
plain `triton.language.extra.libdevice`, identical on all four. It is a bare re-export, so it
carries no Gluon layout handling — the boundary rule for naming plain `tl.*` symbols inside a
`@gluon.jit` body applies to it unchanged, and is written up in `imports-and-launching.md`.

## Entries that belong to another page

These three appear in API-surface discussions often enough to look like appendix material. They are
not — each has a mechanism page that owns it, and the appendix's job is to route, not to summarize.

| Name | Owner |
| --- | --- |
| `atomic_scatter_add` and its `_max` / `_min` / `_and` / `_or` / `_xor` / `_xchg` siblings | `smem-lds-reference.md ## Atomic RMW in LDS — the atomic_scatter_* family` for the signature, the barrier discipline and when a reduction is the right instrument instead; `../workloads/moe.md ## Version gates` for the gate row. LDS atomics on a shared descriptor, 3.8.0 only. `add` and `xchg` take integer or floating dtypes; the other five are integer-only. **There is no pointer-form `smem.atomic_add`** — the scatter spelling is the whole surface, which is why a search for the wrong name reads as absence. |
| descriptor `._reinterpret` / `.gather` / `.scatter` | `smem-lds-reference.md ## Indexed access to LDS — .gather / .scatter` |
| `enable_fp_fusion` | Not a `gl` API at all — a compile option on both the AMD and NVIDIA backends, on by default, and module-wide rather than per-op. Its diagnostic use is in `../workloads/reduction-elementwise.md ## Stage 3 — dequant + residual + norm`. |

## A naming trap: `gl.amd.cdna5`

`gl.amd.cdna5` appears on **3.8.0 only**, and it is **not a gfx950 successor**. Its module body is a
star re-export of `gl.amd.gfx1250` and says so in its own first line. gfx1250 is the separate
data-center fork `index.md` already routes away from; nothing under `cdna5` targets gfx942 or
gfx950. The CDNA namespaces that do are `cdna3` and `cdna4`, both present on all four versions.

The general form of this trap is already recorded — namespace is not architecture support, and
`../hardware/capability-matrix.md` is the authority. `cdna5` is the sharpest instance of it, because
the name implies a generation step that the module does not make.

## What upstream cannot tell you here

Upstream ships **no gfx950 or gfx942 Gluon example kernel** on any of the four versions. So for
every entry on this page the API's *existence and signature* are source-proven, while any claim
about when to reach for it on CDNA is this pack's reasoning and carries no upstream endorsement.
Where a mechanism page disagrees with an appendix line, the mechanism page wins.

**Do not read that as "the AMD Gluon examples are the NVIDIA ones", because there is an AMD example
directory and it is the trap.** `python/tutorials/gluon/` and `python/examples/gluon/` are NVIDIA
(`tcgen05` / TMA / `wgmma` / `two_cta`), but `third_party/amd/python/examples/gluon/` is
AMD-authored and non-empty on **every** one of the four versions — 4 kernels on 3.6.0, 7 on 3.7.0
and 3.7.1, 8 on 3.8.0. Every one of them is named `*_gfx1250.py` and targets gfx1250, so the whole
surrounding kernel is wave32 + `AMDWMMALayout` + TDM + `mbarrier` + multi-CTA, none of which exists
on gfx942 or gfx950 (`../hardware/capability-matrix.md`). The directory therefore reads as
"official AMD usage" and is not official usage *for this pack's targets*: `AMDMFMALayout`,
`buffer_load_to_shared`, `mfma_scaled`, `gl.amd.slice` and `inline_asm_elementwise` appear **zero**
times across all eight of them. This is the same trap as `gl.amd.cdna5` above — a path component
saying `amd` is not architecture support — and it is better disguised, because the path is real and
the code runs.

Where CDNA3/CDNA4 Gluon *is* exercised upstream is the tests: `python/test/gluon/test_frontend.py`
parameterizes on `HIP_TARGET_CDNA3` (gfx942) and `HIP_TARGET_CDNA4` (gfx950) for the MFMA layouts,
`mfma` / `mfma_scaled`, `scaled_upcast`, `PaddedSharedLayout`, `buffer_atomic_rmw`, the CDNA4
async-copy entry points and `warp_pipeline_stage`. Read those for *existence and spelling on this
target*. Do not read them for how to compose a kernel: their bodies are degenerate by design
(`x = i + one`), because what they assert is the emitted IR, not a schedule.
