# Capability matrix (gfx950 / gfx942)

What the Gluon / Triton **3.8.0** surface reaches per target (gfx950 = CDNA4 main line, gfx942 =
CDNA3 downgrade). ISA-level facts: `isa-mechanisms.md`; gfx942 deltas: `cdna3-gfx942.md`; SKU
peaks: `extract_sku.py --facts <SKU>`. A `gl.amd.cdna4.*` symbol compiling on gfx942 does not prove
correctness there; a gfx942 blocker does not reject a gfx950 path. Availability = what compiles
and verifies on the installed build; probe (compile + correctness + read the ISA) before trusting a
cell. RDNA / gfx1250 rows are out of scope.

## Lever availability by layer (Gluon)

Legend: ✓ supported · ◐ hand-written / partial · — not on arch · blocked = API/version.

| Lever | gfx950 | gfx942 | Notes |
| --- | --- | --- | --- |
| `gl.load` / `buffer_load` | ✓ | ✓ | `buffer_load` offsets are **element** offsets (emitter scales by element bytes); `cdna4` re-exports the `cdna3` op |
| `buffer_load_to_shared` (async) | ✓ | ◐ | gfx950 128- or 32-bit/lane, the default authored-ring producer; gfx942 via `gl.amd.cdna4.async_copy` (no `cdna3` async submodule; the generic `ttg.async_copy_global_to_local`, e.g. `nvidia.ampere.async_copy.async_load`, also lowers there to `global_load_lds_dword`), **32-bit per lane only** (128-bit fails `failed to translate module to LLVM IR`); destination unswizzled or swizzled, its `order` matching the source layout's fast dim (`[1, 0]` for a row-major tile; a mismatch fails); A/B vs sync staging. See `## Direct-to-LDS granularity (per arch)` |
| LDS swizzle / padded shared | ✓ | ✓ | padded async dest: `../gluon/memory-reference.md` |
| `ds_read_tr` transpose | ◐ (no Gluon source API) | — | |
| hand-written pipeline (register prefetch -> authored LDS ring -> `warp_pipeline_stage`) | ◐ | ◐ | **`num_stages` is consumed by no Gluon pass in 3.8.0**; `warp_pipeline_stage` needs `num_warps>=8` (2 waves/SIMD) (`../tile-programming/pipeline.md`) |
| re-injected plain pipeliner (`gluon_swp` / `patch_reinject`) | ◐ diagnostic | ◐ diagnostic (sync staging) | sizes `lost_pipeline` debt; numbers labelled `injected`, not counted as a win by policy (`../method/recover.md`) |
| stock `coexec` scheduler strategy | blocked | blocked | 3.8.0: `is_coexec_scheduler_supported(arch)` = `arch == "gfx1250"`. `TRITON_HIP_USE_COEXEC_SCHEDULER` is read nowhere. `llvm_fn_attrs` `amdgpu-sched-strategy=coexec` on gfx950/942 compiles with LLVM warning "'coexec' is only supported for gfx1250" and runs the default (the ISA still changes - an asm diff is not evidence) |
| `llvm_fn_attrs` sched strategy | ✓ (3.8.0 only) | ✓ | accepted: `max-ilp`, `max-memory-clause`, `iterative-ilp`, `iterative-minreg`, `iterative-maxocc`; unknown values (`max-occupancy`, `iterative-occupancy`) are **silently ignored**. See `## Memory and scheduling matrix` |
| GEMM AGPR / compiler interleave | ◐ GEMM-only | blocked | AGPR half via `llvm_fn_attrs` (row below); not on VALU-between-matmul |
| LLIR schedule co-design (authored pass) | ◐ | blocked by cost model | version-dependent (not stock 3.8.0): the gfx950 tutorial LLIR scheduler plugin v1.0-v2.3 has no `16x16x16` / `32x32x8` rows and skips those regions silently (source-verified); the v0.1 build scheduled an unmodelled shape rather than skipping it; v3.0 has the rows but an arch gate skips gfx942; Triton main from gfx950-tutorial-v3.1 schedules gfx942 regions (compile-verified) (`cdna3-gfx942.md ## Pipeline / scheduling`; `../tile-programming/llir-codesign.md`) |
| tile slicing / `convert_layout` | ✓ | ✓ | |
| regular `cdna*.mfma` | ✓ | ✓ | dtype/shape-specific, `## Matrix operation matrix` |
| `cdna4.mfma_scaled` | ✓ | — | fp4 / fp8 at K=128/64 |
| plain `tl.dot` / `tl.dot_scaled` | ✓ | ✓ | comparator |
| `gl.inline_asm_elementwise` | ✓ | ✓ | on 3.6.0 / 3.7.x / 3.8.0; elementwise only (tensors in, tensors out), cannot enclose Gluon code (`../gluon/inline-asm-reference.md`) |

## Direct-to-LDS granularity (per arch)

Backend `TargetFeatures.cpp::supportsDirectToLdsLoadBitWidth` (source-verified 3.8.0):

| Arch | widths | Consequence |
| --- | --- | --- |
| `gfx950` | **{128, 32}** | a **set, not a floor**: 64 bit (8 B/lane) is rejected although > 32. Land `size_per_thread x bits` **on** a member (upstream: *"size per thread * bits per element must be 128 or 32"*) (`../gluon/memory-reference.md ## Minimum per-thread granularity`) |
| `gfx942` | **{32}** | narrowest; often `s_waitcnt`-bound - A/B vs sync staging |

**Coverage rule:** the fast (contiguous) dim must be **exactly covered** by `threads_per_warp *
size_per_thread` (no replication); a sub-floor or replicated fast dim is rejected at lowering. It is
a legality test, not a transaction-width test (`../tile-programming/memory-path.md`). Failure
strings for a wrong width: `failed to translate module to LLVM IR` (`unrealized_conversion_cast`)
or `failed to legalize operation 'ttg.async_copy_global_to_local'`.

## Matrix operation matrix

| Target | Op path | Dtype | Status | Note |
| --- | --- | --- | --- | --- |
| gfx950 | `AMDMFMALayout(v4)` + `cdna4.mfma` | bf16/fp16 -> fp32 | supported | verify K width, result layout, store conversion |
| gfx950 | `cdna4.mfma` | int8 -> int32 | supported | int8 -> **fp32** acc is an API blocker (Gluon MFMA expects int32 acc; plain `tl.dot` inserts i32->f32) |
| gfx950 | regular `cdna4.mfma` | fp8 e4m3fn at `[16,16,32]` / `[32,32,16]` | supported (source) | **registry is keyed per `(version, M, N, K)`**: these two shapes only |
| gfx950 | regular `cdna4.mfma` | fp4 (any), fp8 at other K | blocked | error `no matching matrix core intrinsic ... f8E4M3FN` = one shape missing, not the dtype; use `mfma_scaled` |
| gfx950 | `cdna4.mfma_scaled` | fp8 e4m3/e5m2, fp4 e2m1 | supported (source) | fp8 lands on `[16,16,128]` / `[32,32,64]`; no scales -> `mfma_scaled(a, None, "e4m3", b, None, "e4m3", acc)` (unit e8m0 materialized). Plan operand + scale layouts, K width, acc and store dtype together |
| gfx950 | plain `tl.dot` / `tl.dot_scaled` | fp8 / scaled | supported | plain `tl.dot(fp8)` -> scale-less `tt.dot_scaled` -> `v_mfma_f32_32x32x64_f8f6f4`; `matrix_instr_nonkdim` is version-sensitive |
| gfx950 | `cdna4.scaled_upcast` | fp8 e4m3 + e8m0 -> bf16 | GPU-checked (3.8.0: bit-exact vs torch with the raw E8M0 expanded to the source shape; `v_cvt_scalef32_pk_bf16_fp8`) | Gluon-only one-op dequant (no plain-`tl` standalone equivalent); keeps a bf16 matmul (`../tile-programming/low-precision.md`) |
| gfx950 | `float8_e4m3fnuz` / `float8e4b8` (and `float8e5b16`) | gfx942-style fp8 | trap | silently **upcast to fp16** (warning only; plain `tl.dot` and Gluon `cdna4.mfma` alike, compile-checked 3.8.0) - the kernel is no longer fp8; use OCP fp8 |
| gfx942 | plain `tl.dot` | fp8 | supported (FNUZ) | `tl.dot(fp8e4b8)` is native `v_mfma_f32_32x32x16_fp8_fp8`; `tl.dot(fp8e4nv)` (OCP) is **silently upcast** to fp16 (`v_mfma_f32_32x32x8_f16`, no warning) |
| gfx942 | `gl.amd.cdna3.mfma` (the same function as `cdna4.mfma`) + `AMDMFMALayout(version=3)` | FNUZ fp8 `float8e4b8` / `float8e5b16` | supported | `[16,16,32]` / `[32,32,16]` lower to `v_mfma_f32_*_fp8_fp8` / `_bf8_bf8` (compile-verified 3.8.0; numerics not GPU-checked). OCP `float8e4nv` with `version=3` fails (`no matching matrix core intrinsic ... f8E4M3FN`) |
| gfx942 | `version=4` layout + OCP `float8e4nv` | fp8 | **wrong-result** | compiles to the same `v_mfma_f32_*_fp8_fp8`, which gfx942 reads as FNUZ; `version=4` + FNUZ fails. Convert OCP producers to FNUZ |
| gfx942 | `AMDMFMALayout(v3)` + MFMA | bf16/fp16 | supported shapes `16x16x16`, `32x32x8` | `16x16x32` bf16 fails the pass manager (compile-verified 3.8.0) |
| gfx942 | `AMDMFMALayout(v3)` + MFMA | int8 | supported | verify acc / store dtype |
| gfx942 | `mfma_scaled` | fp8/fp4/MX | — (cannot-select) | no scaled MFMA in the CDNA3 ISA |

## Memory and scheduling matrix

| Target | Mechanism | Status | Notes |
| --- | --- | --- | --- |
| gfx950 | `buffer_load_to_shared` offsets | version-sensitive | needs 32-bit offsets; which offset layout families lower (`Blocked`/`Slice` vs `DistributedLinearLayout`) has changed across Triton releases (on 3.8.0 both an exact-cover `Blocked` and an `offset_bases`-derived `DistributedLinearLayout` producer lower; other minors not re-checked on gfx950) - keep a sync `convert_layout` fallback and probe per build |
| gfx950 | `convert_layout`, `gl.reshape` / `gl.split` | supported | reshape/split may infer `DistributedLinearLayout`; stores often need explicit `convert_layout`; hoist converts out of hot loops |
| both | `sched_barrier` / `sched_group_barrier` / `iglp_opt` | no user surface on 3.8.0 (blocked on some 3.7.0) | `warp_pipeline_stage` markers emit `s_setprio` + `sched_barrier` for you (`../tile-programming/warp-pipeline.md`) |
| gfx950 | `amdgpu-agpr-alloc` via `llvm_fn_attrs` | 3.8.0 only | `llvm_fn_attrs=[["amdgpu-agpr-alloc", "256"]]` works **only at 1 wave/SIMD**: at 2+ waves `"256"` exceeds the per-wave budget - LLVM "no registers from class available to allocate" and the returned binary fails at launch (`hipErrorLaunchOutOfResources` or a memory fault). Use the pair form for `min,max` (the string form splits on commas). GEMM-class only; verify accumulator in `a[..]` in the asm (`../tile-programming/llvm-fn-attrs.md`) |
| gfx950 | `amdgpu-mfma-vgpr-form` as an `llvm_fn_attrs` entry | inert | it is an `llc` option, not a function attribute (identical asm for 0/1) |
| gfx950 | `llvm_fn_attrs` itself | 3.8.0 only | absent on 3.6.0 / 3.7.0 / 3.7.1 (unrecognized option). Open pass-through: any `name=value` is applied to the kernel function; per-compile, also on plain `@triton.jit`. Measured gains came on kernels with schedulable ILP, and different strategies paid on different kernels; on the serial chain-recurrence kernels measured, every strategy tried was neutral-to-negative - sweep + A/B, accept on an asm diff (`../tile-programming/llvm-fn-attrs.md`) |
| gfx950 | `LLVM_PASS_PLUGIN_PATH` (out-of-tree pass) | host-build-gated | read in `make_llir` and reaches Gluon, but needs a default-visibility host build (`TRITON_EXT_ENABLED`, off) - not reachable on the stock 3.8.0 wheel (`../tile-programming/llir-codesign.md`) |
| gfx950 | `gl.warp_predicate` (per-wave branch) | **absent upstream** (vendor fork only; absent in the 3.8.0 wheel, checked) | no `warp_id` / `lane_id` scalar on 3.6-3.8 (3.8.0 checked); an `if` takes only a 0-d (CTA-uniform) condition; inline asm cannot wrap Gluon code (only a whole region written in asm can use exec per wave) |
| gfx950 | 2D grid + `num_warps=1` | measured, shape-scoped (Triton 3.8.0) | observed upstream: a small kernel on a multi-block 2D grid ran much slower than flattened 1D. Reproduced only with a small axis-0 / large axis-1 grid on dispatch-rate-bound tiny programs, at any `num_warps`; the large extent on axis 0 (or 1D) removed it, and grid rank alone tied (`../workloads/prologue.md ## A 2D grid with a small axis 0: grid shape, not num_warps=1`) |
| both | producer/consumer warp specialization (NVIDIA form) | arch-gated: the NVIDIA recipe does not port as written | it relies on named-barrier warpgroups + dynamic register repartition, which CDNA3/CDNA4 lack (static per-wave allocation); wave role splits with ordinary barriers / flags remain possible |
| gfx942 | `cdna3.buffer_load` / `buffer_store` | supported | element offsets + typed `other` |
| gfx942 | `cdna4.async_copy.buffer_load_to_shared` + `commit_group` / `wait_group` + `load_shared_relaxed` | supported | 32-bit; some transpose / dot-operand offset layouts under v3 + `SwizzledSharedLayout` have failed LLVM translation (`unrealized_conversion_cast`; observed, not reproduced as a specific kernel) - probe per kernel, fall back to `PaddedSharedLayout` + `DistributedLinearLayout`; `compute_efficient_padded_shared_layout` asserts an MFMA v4 parent (3.8.0 source), so derive the padded layout another way on gfx942 |

## Instruction shape rules

- Triton >= 3.6 CDNA: `instr_shape` is 3D `[M, N, K]` (old 2D `[32, 32]` examples are stale).
- gfx950: bf16/fp16 `[16,16,32]` / `[32,32,16]`; regular fp8/bf8 only `[16,16,32]` / `[32,32,16]`;
  scaled `[16,16,128]` / `[32,32,64]`; int8 `[16,16,32]` / `[32,32,16]` and the CDNA4 `[16,16,64]` /
  `[32,32,32]` (`v_mfma_i32_16x16x64_i8`, `v_mfma_i32_32x32x32_i8`; compile-verified 3.8.0).
- gfx942: bf16/fp16 `[16,16,16]` / `[32,32,8]`; int8 `[16,16,32]` / `[32,32,16]`; fp8 (FNUZ only)
  `[16,16,32]` / `[32,32,16]` with `version=3`.

## Tile / shape constraints (Gluon language)

Tensor shape elements (incl. `gl.arange`) must be a **power of 2** (3.8.0, compile-checked): `gl.arange(0, 192)` -> "Shape
element 0 must be a power of 2". Shared-memory allocations and `.slice`s obey the same rule and raise the same
catchable compile error (compile-checked 3.6.0-3.8.0). Pad (`D=192 -> 256`) or split (`128 + 64`).

## Deprecated / version-sensitive knobs

`num_stages` on the Gluon path: inert in 3.8.0 (no Gluon pass reads it; a compile-only loop check gave identical ISA for 1 / 4 / default). Triton 3.7.0 `kpack`: overwritten
to 1. `matrix_instr_nonkdim`: active, re-sweep across versions.

## Build switch permission

The build is fixed: compiler-pass edits / rebuilds are not reachable on the stock 3.8.0 wheel - a
missing feature is a scoped ceiling, not a fix (`../tile-programming/compiler-contract.md`).

## Reading `hw_constants.json` across generations

Pass `--arch` to every tool; a key absent on one arch is not permission to use the other's value:

- `fp8_dtype`: gfx942 is FNUZ; on gfx950 the FNUZ spelling silently upcasts to fp16 (above).
- LDS allocation granule: `lds_alloc_granule_bytes` on both arches (gfx942 512, gfx950 1280; the older
  `lds_min_alloc_bytes` / `lds_align_bytes` keys carry the same values; 320 dwords, LLVM
  `getLDSAllocGranule(GK_GFX950) = 1280`). For
  LDS-limited occupancy round each workgroup's LDS up to that multiple before dividing 160 KiB (64 KiB).
- A config that dies on gfx942 with `OutOfResources: ... Hardware limit: 65536` can run on gfx950
  (2.5x LDS) and be slower: look for `spill=` / `probe.py measure`, not an exception.
- `ds_read_b128_2way_stride_bytes` exists only on CDNA4: a swizzle conflict-free on gfx942 is not
  known to stay so; re-derive.
- `matrix_layout_family` is `version=3` vs `version=4`: layout digests should differ across
  generations.
