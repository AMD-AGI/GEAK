# Capability Matrix (gfx950 / gfx942)

**Single source of truth** for target-, dtype-, and matrix-op capability on
gfx950 (CDNA4, the main line) and gfx942 (CDNA3, the downgrade comparison — every table lists the
gfx950 cell first). Spellings follow upstream Triton **3.8.0**; 3.6 / 3.7 differences appear only as
version-scoped rows or notes. Entry: `atlas.md`. **SKU peaks** (ridge, CUs, L2):
`amd-cdna4-skus.md`, `amd-cdna3-skus.md`, `amd-rdna3-skus.md`, `amd-rdna35-skus.md`, `amd-rdna4-skus.md` —
same ISA, different chips (e.g. MI355X vs MI350X, MI300X vs MI308X); machine source
`perf_knowledge/hardware/data/sku.json`. The Gluon API files
(`../gluon/*`) link here instead of repeating support claims. The ISA-level (not API-level)
cross-generation MFMA table is GEAK's `perf_knowledge/hardware/shared/matrix_core_mfma_smfmac.md`
("Cross-generation MFMA capability matrix"); this page records what the **Gluon / Triton 3.8.0
surface** reaches on top of it.
RDNA / WMMA: `rdna-fork.md`. **gfx1250 (CDNA5 / MI450)**: WMMA+TDM sub-target — see gfx1250 rows and
`../gluon/atoms-reference.md`. gfx1250 is a separate data-center ISA fork, not RDNA4
(RDNA4 client = gfx1201, R9700 / RX9070 XT, occupancy-first WMMA, no TDM).
**RDNA4 client (gfx120* / gfx1201)** lever rows are included below (ISA + code anchors
from `rdna-fork.md`); most still need a **compile-verified probe on gfx1201** — only
paths backed by ISA docs + a code anchor are marked `source-proven`, the rest are
`unknown-needs-probe`.

**Role in the three-table split (avoid drift):** this file = lever **availability** /
arch gate (✓ / ◐ / blocked per target). *Which* lever per bound class + *when it applies*
(gate / cost / expected-move / verify) live in `bound-class-signals.md` (lever cards +
gating laws); the *primary metric* per class lives in `../method/profile.md`.

## Bound class → primary lever

Consult after `bound-class-signals.md` confirms the class (its `## Bound-class decision
tree` → leaves), then check availability here, then apply gate/cost from its `## Lever
cards`.

| Bound class | Primary lever (first) | Layer |
| --- | --- | --- |
| memory / bandwidth | `cut_hbm_bytes`: narrow dtype / fuse streaming pass / reuse (in-flight inert) | 2 |
| memory / latency | `hide_mem_latency`: more outstanding loads / deeper prefetch (occupancy-gated) | 2/4 |
| memory / l2-locality | `GROUP_SIZE_M` / XCD PID remap | 6 |
| LDS-bound | swizzle / padding / `ds_read_tr` (if available) | 3 |
| pipeline / latency | hand-written overlap in `../tile-programming/pipeline.md` order: register prefetch → authored LDS ring (`commit_group`/`wait_group`) → `warp_pipeline_stage`; occupancy | 4 |
| latency / ifetch | `reduce_icache_pressure`: less unroll / fewer jumps | 7 |
| register / occupancy | tile slicing, LDS dedup | 5 |
| MFMA-issue / schedule | scheduling-model choice: `gemm_compiler_stack` (`compiler_interleave`) vs `warp_pipeline_schedule` (`inter_wave`, wave-ping-pong) — and `authored_stage` is the stock base both sit on | 1.5/4 |
| MFMA-issue / tile | tile shape, `instr_shape`, MFMA layout chain | 7 |
| compute / divergence | `reduce_divergence`: mask elision / uniform-ize / pow2-align | 7 |
| VALU / fused epilogue | fold scalars, `scaled_upcast`, restructure | 7 |

**Memory sub-resource split (NCU Memory Workload Analysis parity):** memory-bound is not one lever —
`bandwidth` (cut bytes), `latency` (add in-flight), `l2-locality` (remap) need OPPOSITE actions.
`bound-class-signals.md ## Bound-class decision tree (exhaustive, ordered, defaulted)` + the thresholds (`hbm_bw_saturated_pct`,
`l2_hit_locality_pct`, per-arch `hw_constants.l2_fabric_latency_high_cyc`) route them.

**Arch default = probe, not yes (gfx1250 / RDNA).** A lever card that does not list an arch defaults to
`probe` in `classify._pick_levers` (never silently `yes`). **gfx1250 (CDNA5/MI450) and gfx1201/gfx1100
(RDNA) are WMMA/TDM, not MFMA** — any `isa_gate: MFMA/scaled-MFMA` card is additionally dropped when
`context.isa.mfma/scaled_mfma` is false, so MFMA-only levers never leak onto a WMMA arch; the WMMA
equivalent is probe/author (see the gfx1201/gfx1250 rows in the matrices below).

## Lever availability by layer (Gluon)

Legend: **✓** supported · **◐** hand-written / partial · **—** not on arch · **blocked** API/version.

| Lever | Layer | gfx950 | gfx942 | Notes |
| --- | --- | --- | --- | --- |
| `gl.load` / buffer_load | 2 | ✓ | ✓ | memory anchor |
| `buffer_load_to_shared` / async | 2 | ✓/◐ | ◐ | gfx950: 128-bit (16 B/thread) or 32-bit direct-to-LDS, the default authored-ring producer (`async_copy` + `commit_group`/`wait_group`); build the destination **swizzled** first — a padded async destination has failed LLVM translation. gfx942 downgrade: via the **`cdna4.async_copy`** entry (the `cdna3` namespace has no async submodule; the generic `ttg.async_copy_global_to_local` is illegal on gfx942), 32-bit only, swizzled destination `order=[1,0]`, version-sensitive offset layouts — sync staging is the default ring there (see `## Direct-to-LDS granularity (per arch)`) |
| LDS swizzle / padded shared | 3 | ✓ | ✓ | |
| `ds_read_tr` transpose | 3 | ◐/Tier-B | — | no Gluon source API |
| scheduling-model choice | 1.5 | ✓ | ✓ | `authored_stage` / `compiler_interleave` / `inter_wave`; `../tile-programming/scheduling-model.md` |
| `warp_pipeline_schedule` (inter_wave) | 4 | ✓ | ✓ | stock Triton, no plugins/env; needs 2 waves/SIMD |
| hand-written pipeline (register prefetch → authored LDS ring → `warp_pipeline_stage`) | 4 | ◐ | ◐ | the climb default, in the order `../tile-programming/pipeline.md` defines; `num_stages` is consumed by no Gluon pass in 3.8.0 (budget parameter / champion record only). `warp_pipeline_stage` gated on `num_warps>=8` |
| re-injected plain pipeliner (`gluon_swp` / `patch_reinject` / `patch_async_reinject`) | 4 | ◐ diagnostic | ◐ diagnostic (sync staging) | **lowest rung**: a below-parity diagnostic for `lost_pipeline`, or a last resort when the hand-written ring cannot reach parity; numbers labelled `injected`, never a win, never on an incumbent Gluon kernel (`../method/recover.md`) |
| stock coexec scheduler strategy (`TRITON_HIP_USE_COEXEC_SCHEDULER`) | 4 | opt-in | opt-in | upstream 3.8.0: `is_coexec_scheduler_enabled` = the env var if set, else `arch == gfx1250` — so **off by default on gfx950 and gfx942**. Opt in process-wide with `TRITON_HIP_USE_COEXEC_SCHEDULER=1` (the backend then adds `amdgpu-sched-strategy=coexec` only at `num_warps <= 4`) or per compile with `llvm_fn_attrs=[["amdgpu-sched-strategy","coexec"]]` (any `num_warps`); accept on an assembly diff. Try it before authoring a matrix+VALU schedule (`../tile-programming/compiler-contract.md ## What upstream 3.8.0 actually gives you`) |
| `gemm_compiler_stack` (`compiler_interleave`) | 4 | ◐ GEMM-only | blocked | upstream 3.8.0 reaches the AGPR half per compile (`llvm_fn_attrs`) and the interleave only from an authored pass; no post-assembly stage exists. Not on VALU-between-matmul |
| LLIR schedule co-design (throughput + co-execution models) | 4 | ◐ author-a-pass | target-specific-blocker | **gfx942 blocker is a pass's per-shape cost model, not silicon**: CDNA4-developed tables lack `16x16x16`/`32x32x8`, so every region is skipped silently, and recalibrating needs the window model too (`cdna3-gfx942.md ## Pipeline / scheduling`). Availability on gfx950 is per build — three states, two of which fail quietly (`../tile-programming/llir-codesign.md ## The plugin tier`) |
| tile slicing / `convert_layout` | 5 | ✓ | ✓ | |
| regular `cdna*.mfma` | 7 | ✓ | ✓ | dtype-specific |
| `cdna4.mfma_scaled` | 7 | ✓ | — | gfx950 fp8/fp4 Gluon path |
| plain `tl.dot` / `tl.dot_scaled` | 7 | ✓ | ✓ | comparator |
| `gl.inline_asm_elementwise` (ISA escape) | all | ✓ | ✓ | **present on all four versions checked** (3.6.0 / 3.7.0 / 3.7.1 / 3.8.0), re-exported from Triton core — so an "absent instruction" here is almost never a language ceiling. Elementwise-only: it takes tensors and returns tensors and cannot enclose Gluon-level code (see the `gl.warp_predicate` row). Gate it before reaching: the block is opaque to the optimizer and durability is per-class. Gluon pack: `../gluon/inline-asm-reference.md ## Classifying a site: mechanism × intent` |

Full dtype×API evidence: §Evidence detail below.

## Direct-to-LDS granularity (per arch)

Backend fact (`TargetFeatures.cpp::supportsDirectToLdsLoadBitWidth`) — the set of
direct-to-LDS load widths the async copy can emit, per arch:

| Arch | Family | direct-to-LDS widths | per-thread chunk | Consequence |
| --- | --- | --- | --- | --- |
| `gfx950` | CDNA4 | {128, 32} | 128-bit (16 B) **or** 32-bit (4 B) | wide async DMA. **This is a set, not a floor** — 8 B/thread is above 4 B and is still rejected; land the per-thread byte count **on** a member (`../gluon/memory-reference.md ## Minimum per-thread granularity`) |
| `gfx942` | CDNA3 (downgrade) | **{32}** | 32-bit (4 B / 2×bf16) | narrowest async DMA; relieves the feed / arch-VGPR but tends to become `s_waitcnt`-bound — measure vs sync staging, do not assume a win |
| `gfx1250` | CDNA5 | {128, 64, 32} | up to 128-bit + 64-bit | widest; TDM sub-target |

**Coverage rule (non-GFX1250):** the fast (contiguous) dim must be **exactly
covered** by `threads_per_warp * size_per_thread` — no replication. A sub-floor or
replicated fast dim is rejected at lowering (a layout/granularity constraint, not a
build ceiling). **This rule is a legality test, not a width test** — it constrains
the product over one group, and the transaction a load produces is set by the
contiguous run within a group, which the group count does not enter. A layout can
cover the tile exactly and still halve the run
(`../tile-programming/memory-path.md ## A load's width is issue-side; a
transaction's width is access-side`). gfx942 async copy is reached via the **`gl.amd.cdna4.async_copy`**
entry (the `cdna3` namespace has no async submodule); the generic
`ttg.async_copy_global_to_local` is illegal on the gfx942 backend. Full CDNA3 page:
`cdna3-gfx942.md`.

## Evidence detail

Status meanings:

| Status | Meaning |
| --- | --- |
| `supported` | Correctness proven for the stated target, dtype, and API path. |
| `source-proven` | Local source exposes the path; still needs a smoke + correctness probe. |
| `compile-only` | Compiles, but correctness not proven. |
| `wrong-result` | Executes but produces incorrect values. |
| `version/API-blocker` | The installed Triton/Gluon API or lowering stack blocks the path. |
| `unknown-needs-probe` | Evidence missing; not support or rejection. |
| `target-specific-blocker` | Blocker scoped to the named target; do not generalize. |
| `performance-hazard` | Compiles/runs but local evidence shows likely slow/unstable; require a timing gate. |

Evidence: `local-confirmed` (local compile/correctness/runtime), `local-source-proven`
(operator-local or Triton/Gluon source), `upstream-doc-proven`, `unknown-needs-probe`.

Upstream Triton 3.8.0 is the reference version for every cell (rows marked `v3.8.0` were read at
that tag). When feedback mentions Triton 3.7.0, treat it as a downgrade note and prefer the intended
`triton-3.7.0+rocm7.2` build; do not promote blockers from weaker/older 3.7.0 builds into generic
blockers. Record the installed Triton/ROCm identity and run local smoke +
correctness + perf probes before trusting a path.

## Target family summary

| Target | Family | Wave | Layout family | First safe model | Do not infer |
| --- | --- | --- | --- | --- | --- |
| `gfx950` | CDNA4 | wave64 | `AMDMFMALayout(version=4)` | CDNA-style layouts + target/version checks, regular MFMA anchor before scaled MFMA | gfx942 blockers, or correctness without local probes |
| `gfx942` | CDNA3 | wave64 | `AMDMFMALayout(version=3)` | Blocked layouts, generic `gl.load/store`, regular MFMA after matrix-plan checks | CDNA4 scaled MFMA or gfx950 FP8 behavior |
| `gfx1201` | RDNA4 (client) | **wave32** | `AMDWMMALayout` (v8-operand) | WMMA `_w32` after matrix-plan checks; wave32 occupancy (≤256 VGPR/wave), hand LDS ping-pong (no `num_stages`) | CDNA MFMA / `mfma_scaled`, gfx1250 TDM / named-barrier, or wave64 occupancy formulas |

Namespace is not architecture support: a `gl.amd.cdna4.*` symbol compiling on
`gfx942` does not prove correctness for `gfx942`; a `gfx942` blocker does not
reject a real `gfx950` path.

## Gluon capability levels

| Level | Meaning | Required evidence |
| --- | --- | --- |
| 0 | `triton.experimental.gluon` imports | import path + version recorded |
| 1 | 1D `BlockedLayout` load/store executes | minimal `@gluon.jit` smoke + correctness |
| 2 | target layout objects construct | layout object creation outside JIT body |
| 3 | 2D blocked load/store executes | correctness on the needed rank/layout |
| 4 | `DotOperandLayout` + MFMA executes | matrix correctness for dtype/accumulator |
| 5 | full pattern executes (dot + reduction + dot) | measured boundary path feeds final output |

Level 0/1 is not evidence for Level 4/5.

## Matrix operation matrix

| Target | API / op path | Dtype/path | Status | Evidence | Required next check |
| --- | --- | --- | --- | --- | --- |
| `gfx950` | `AMDMFMALayout(v4)` + `cdna4.mfma` | INT8 -> int32 acc | `supported` | `local-confirmed` | Verify accumulator dtype + store conversion. |
| `gfx950` | `AMDMFMALayout(v4)` + `cdna4.mfma` | INT8 -> float32 acc | `version/API-blocker` | `local-confirmed` | Direct Gluon MFMA expects int32 acc; plain `tl.dot` may insert i32->f32. |
| `gfx950` | `AMDMFMALayout(v4)` + `cdna4.mfma` | BF16/FP16 -> fp32 acc | `supported` | `local-confirmed` | Verify K width, result layout, store conversion. |
| `gfx950` | `AMDMFMALayout(v4)` + regular `cdna4.mfma` | FP8 e4m3fn at `[16,16,32]` / `[32,32,16]` | `supported` | `local-source-proven` | **The registry is keyed per `(version, M, N, K)`, not per dtype.** `v3.8.0`'s MFMA v4 table registers regular `mfma_f32_16x16x32_fp8_fp8` and `mfma_f32_32x32x16_fp8_fp8` (with the fp8/bf8 mixes), so these are reachable from plain `cdna4.mfma`. |
| `gfx950` | `AMDMFMALayout(v4)` + regular `cdna4.mfma` | FP4 e2m1 (any shape); FP8 at an unregistered shape | `version/API-blocker` | `local-confirmed` | FP4 has no entry in the regular table at all; fp8 at e.g. `[16,16,128]` / `[32,32,64]` has none either. The `no matching matrix core intrinsic ... f8E4M3FN` error is that registry missing **one shape**, not a dtype-level prohibition — look up which `(M, N, K)` is registered before abandoning the dtype. Those cases use `cdna4.mfma_scaled` (next row). Plain `tl.dot(fp8)` is separate (supported, scale-less `tt.dot_scaled`). |
| `gfx950` | `cdna4.mfma_scaled` / scaled MFMA | FP8 e4m3/e5m2, FP4 e2m1 | `supported` | `local-source-proven` | CDNA4-only; the required Gluon matrix path for fp4, and for fp8 at the shapes the regular table does not register (`withScale` maps every f8f6f4 type onto the fp4 key, so fp8 lands on `[16,16,128]` / `[32,32,64]`). No explicit scale -> pass `None` scales + format string (`mfma_scaled(a, None, "e4m3", b, None, "e4m3", acc)`, unit e8m0 scale materialized internally). Plan data operands, scale layouts, K width, acc dtype, store dtype together. |
| `gfx950` | plain `tl.dot` / `tl.dot_scaled` | FP8/scaled | `supported` | `local-confirmed` | First-class comparator; optimal `matrix_instr_nonkdim` is version-sensitive. |
| `gfx950` | `cdna4.scaled_upcast` (standalone fp8 + block-scale -> bf16 fuse) | FP8 e4m3 + e8m0 scale -> bf16 | `source-proven` | `local-source-proven` | Gluon-only ONE-op dequant (fuse fp8->bf16 with the block scale). **No plain-`tl` standalone equivalent** — plain has only scaled-DOT (`tt.dot_scaled`), which forces the low-precision MMA path. Use to remove the manual dequant VALU while KEEPING a bf16 matmul (e.g. when a cosine gate forbids fp8-MMA, `../tile-programming/low-precision.md`). Verify presence per build. |
| `gfx950` | `torch.float8_e4m3fnuz` / fp8e4b8 | gfx942-oriented FP8 | `target-specific-blocker` | `local-confirmed` | Some gfx950 stacks upcast this to FP16; prefer OCP FP8 for native gfx950 FP8. |
| `gfx942` | plain Triton `tl.dot` | FP8 e4m3fn | `supported` | `local-confirmed` | Keep as first-class FP8 GEMM comparator. |
| `gfx942` | `gl.amd.cdna3.mfma` | FP8 e4m3fn | `version/API-blocker` | `local-confirmed` | Do not use for gfx942 FP8; record target-specific blocker. |
| `gfx942` | `gl.amd.cdna4.mfma` | FP8 e4m3fn | `wrong-result` | `local-confirmed` | Compile success without correctness is not support. |
| `gfx942` | `AMDMFMALayout(v3)` + regular MFMA | INT8 | `supported` | `local-confirmed` | Verify output/accumulator dtype + store path per operator. |
| `gfx942` | `AMDMFMALayout(v3)` + regular MFMA | BF16/FP16 | `unknown-needs-probe` | `unknown-needs-probe` | Verify local `instr_shape`, K width, correctness. |
| `gfx942` | `mfma_scaled` / scaled MFMA | FP8/FP4/MX | `target-specific-blocker` | `local-confirmed` | Treat as gfx950-oriented unless local source proves otherwise. |
| `gfx1201` (RDNA4) | `gl.amd.rdna4.wmma` | BF16/FP16 -> fp32 acc | `source-proven` | `upstream-doc-proven` | 16×16×16 WMMA `_w32`, **v8-operand ABI** (different lane map vs RDNA3 v16). Needs compile-verified probe on gfx1201 (`rdna-fork.md`, `../gluon/rdna-wmma-reference.md`). |
| `gfx1201` (RDNA4) | `gl.amd.rdna4.wmma` | FP8/BF8 e4m3/e5m2 -> fp32 acc | `source-proven` | `upstream-doc-proven` | RDNA4-only `V_WMMA_F32_16X16X16_FP8_*`; **no `mfma_scaled` / no blockscale** — FP4 needs dequant+WMMA or IU4 WMMA. Compile-verify per build. |
| `gfx1201` (RDNA4) | SWMMAC 4:2 structured sparse | fp16/bf16/i8 | `unknown-needs-probe` | `unknown-needs-probe` | ISA has SWMMAC (`rdna-fork.md`); Gluon API path unverified — probe before use. |

## Memory and scheduling matrix

| Target | API / mechanism | Status | Evidence | Notes |
| --- | --- | --- | --- | --- |
| `gfx950` | generic `gl.load/store` + explicit `BlockedLayout` | `supported` | `local-confirmed` | First gfx950 memory anchor. |
| `gfx950` | `cdna4.buffer_load` / `buffer_store` | `supported` | `local-confirmed` | Byte offsets, scalar base, layout, fallback/store dtype remain per-operator. |
| `gfx950` | `cdna3.buffer_load/store` namespace | `source-proven` | `local-confirmed` | Equivalent operand types can emit the same ISA as CDNA4; namespace swap is not a direction. |
| `gfx950` | `buffer_load_to_shared` / CDNA4 async copy | `source-proven` | `local-source-proven` | Needs 32-bit offsets; offset tensor often needs an operand-specific distributed layout matching rank/dtype/unit/consumer; a `BlockedLayout` lowering failure is layout-contract evidence first. Also needs the per-thread chunk to land **on** a legal direct-to-LDS width — CDNA4 = {16 B, 4 B}, and it is a **set, not a floor**, so 8 B/thread is rejected even though it exceeds 4 B. Move `size_per_thread` onto a member rather than raising it, or use register staging (`../gluon/memory-reference.md ## Minimum per-thread granularity`). Upstream states the same rule on the op itself: *"size per thread * bits per element must be 128 or 32"*. **Version-gated offset-layout family:** which offset layouts lower for this async copy is **Triton-minor-sensitive** — an older minor may accept only `Blocked` / `Slice` offset layouts and **fail to compile** a `DistributedLinearLayout` async offset, while a newer minor accepts it. **Gate the async path to the build that lowers the needed offset family**, verify per build, and keep a sync (`convert_layout`) fallback for the unsupported minor. |
| `gfx950` | `DistributedLinearLayout` | `supported` | `local-confirmed` | Basis vectors must match tile shape; often the right async-copy offset family. |
| `gfx950` | `SwizzledSharedLayout` | `supported` | `local-confirmed` | Verify shared-memory conversion cost. |
| `gfx950` | `DotOperandLayout` | `supported` | `local-confirmed` | Requires result layout + K-width derivation. |
| `gfx950` | `gl.reshape` / `gl.split` | `supported` | `local-confirmed` | May infer `DistributedLinearLayout`; store paths often need explicit `convert_layout`. |
| `gfx950` | `convert_layout` | `supported` | `local-confirmed` | Supported but can dominate hot loops; hoist or avoid repeats. |
| `gfx950` + Triton 3.7.0 | `sched_barrier` / `sched_group_barrier` | `version/API-blocker` | `local-confirmed` | Version-scoped; on upstream 3.8.0 there is no user-facing surface for these (nor `iglp_opt`) from either tier — record a toolchain ceiling, not a mechanism rejection. **Known workaround: do not hand-issue it — the marker path emits it.** `warp_pipeline_stage` cluster markers lower to `s_setprio` + `sched_barrier` around each cluster, so a kernel that wanted scheduling boundaries has them without naming the mnemonic (canonical: `../tile-programming/warp-pipeline.md`, `../tile-programming/scheduling-model.md`). Reach for the absent symbol only if you need a boundary the markers cannot place. |
| `gfx950` (3.7.0 confirmed; 3.8.0 upstream) | `warp_pipeline_stage` | `supported` | `local-confirmed` | Layer 1.5/3 of the hand-written order; Gate 0 is `num_warps>=8` (`../tile-programming/warp-pipeline.md`). Scheduling hints are not inherently profitable; require quick timing. |
| `gfx950` | AGPR accumulator placement (`amdgpu-agpr-alloc` / `amdgpu-mfma-vgpr-form`) | `source-proven` — **3.8.0 only**, via `llvm_fn_attrs` | `local-source-proven` | These are LLVM **function attributes**, and 3.8.0 passes arbitrary function attributes through to the kernel, so this needs no env var and no fork. GEMM-class only: pure MFMA -> MFMA accumulator chain, accumulator written-only-until-epilogue. On VALU-between-matmul (attention / softmax / MLA / DSA) the accumulator is read-modified every iteration and must stay in VGPR — LLVM's default allocator already makes that split, and forcing the GEMM hint is what breaks it. Verify in the assembly (accumulator in AGPR, fewer in-loop `v_accvgpr_mov`), never from the fact that the option was set. `../tile-programming/compiler-contract.md ## What upstream 3.8.0 actually gives you`. |
| `gfx950` | out-of-tree LLVM pass plugin (region-classifying scheduler you author) | `version/API-blocker` (host-build-gated) | `local-source-proven` | The door is upstream: `LLVM_PASS_PLUGIN_PATH` is read in `make_llir` and reaches the Gluon path. Needs **no LLVM rebuild** but does need a default-visibility host build (`TRITON_EXT_ENABLED`, default OFF) and a binary ABI-matched to the host LLVM revision. Note upstream drops the TargetMachine when a plugin is set, by design, so plugin-on and plugin-off are not the same optimization environment. Four gates, three fail silently — `../tile-programming/llir-codesign.md ## The plugin tier`. A region-classifying pass treats attention as a **target**, not a hazard. |
| `gfx950` | per-wave predicate (`gl.warp_predicate`) | `version/API-blocker` — **absent upstream**, vendor-fork extension | `local-confirmed` | Lowers to `s_and_saveexec` + `s_cbranch_execz`: warp-uniform branch, no cross-wave reduction, no barrier. Required by any "skip the work on the common path" design (lazy softmax rescale); a lane mask is **not** a substitute because the masked multiply still issues. Absent -> record a language ceiling against the build, not a negative result (`../gluon/pipeline-reference.md`). **Known workaround: none *for the thing this extension does*, and specifically not inline asm for it.** The only inline-asm door in Gluon takes tensors and returns tensors, so it cannot enclose a region of **Gluon-level** code you wanted skipped; hand-issuing `s_and_saveexec_b64` around one asm block narrows exec for that block's own instructions and nothing else, which buys the mnemonic without the branch. **Read that at its real width.** A different, shipped shape is not ruled out by it: put the whole region inside the asm text and use exec to hand different waves different roles — roughly a dozen surveyed sites do exactly that, and it is not a substitute for `warp_predicate` but it is not "no workaround" either. Both statements hold at once (`../gluon/inline-asm-reference.md ## Class 4 — synchronization`; in the Gluon pack only). An ordinary `if` does not reach it either: Gluon shares plain Triton's code generator, whose `if` accepts only a 0-d condition, and every route from a tile to 0-d is a reduction across the whole tile — so the condition is CTA-uniform by construction, which is exactly the granularity this extension exists to get below. No per-warp scalar (`warp_id` / `lane_id`) exists in the language surface on any of 3.6.0 / 3.7.0 / 3.7.1 / 3.8.0. |
| `gfx950` | `llvm_fn_attrs="amdgpu-sched-strategy=<s>"` (per-launch portable scheduler) | `source-proven` — **3.8.0 only** | `local-source-proven` | **The option itself is a version gate: `llvm_fn_attrs` does not exist on 3.6.0 / 3.7.0 / 3.7.1** — zero occurrences in the AMD backend on all three — so on those builds the name is an unrecognized compile option, not a scheduler you failed to tune. **The strategy value list is NOT established from source**: LLVM is not vendored in the Triton tree (it is a prebuilt pinned by hash), so the accepted enum cannot be read from what ships. Evidenced inside Triton itself: `iterative-ilp` (upstream's documented example and its test) and `coexec` (set by the backend for gfx1250). The names that circulate for this attribute -- `max-occupancy` / `max-ilp` / `max-memory-clause` / `iterative-minreg` / `iterative-maxocc` -- have zero occurrences in the tree; they may be valid in the pinned LLVM but this pack has not shown it, and since an unrecognized value is silently ignored a flat sweep over them is indistinguishable from them not existing. Diff the generated assembly to confirm a strategy bit. **Adoption evidence, which is not source evidence:** gfx950 kernels do set this attribute -- on plain `@triton.jit` bodies as well as Gluon ones, so it is **not** a Gluon mechanism -- using `iterative-ilp` / `max-ilp` / `iterative-minreg` / `max-memory-clause` / `iterative-occupancy`. What the observed values do **not** give you is an archetype-level rule to inherit: structurally similar kernels disagree, and adjacent launches in one dispatcher take different values — so read the value off the launch you are actually looking at rather than off the workload it belongs to. That establishes adoption, not the enum, so the assembly diff remains the acceptance signal; but three of those four are values the paragraph above lists as unevidenced, so a flat sweep over them earns one static read before being written off. Two properties beyond the strategy list: it is an **open LLVM function-attribute pass-through with no allow-list** — arbitrary `name=value` pairs are parsed and applied to the kernel function, and upstream's own doc example pairs a strategy with `noinline`, so the scheduler is one use and not the boundary; and it is **per-compile**, which is the granularity a process-wide env var cannot express (the same source can select a different attribute per M regime). Unlike the GEMM-only env knobs it applies to VALU-between-matmul too, BUT helps only **ILP-starved** kernels (low occupancy with reorderable slack); on a serial dependency chain with no slack all strategies are **neutral-to-negative** — sweep + A/B by bound class, never default on. **The mechanism chapter is `../tile-programming/llvm-fn-attrs.md`** -- what you may write on the option, the two failure modes, the assembly-diff acceptance procedure, the cost and the downgrade path all live there; `../tile-programming/instruction-scheduling.md ## llvm_fn_attrs — the portable, per-compile scheduler strategy` places it in the layer. |
| `gfx950` | 2D grid + `num_warps=1` | `performance-hazard` | `local-confirmed` | Multi-block 2D grids can be much slower than 1D linear grids; test grid rank + warp count separately. |
| `gfx950` | producer/consumer warp-specialization (warpgroup specialization + dynamic register repartition + named-barrier warpgroups) | `arch-gated` | `compile-verified` | The most arch-sensitive FA3-forward piece: presupposes cheap dynamic register repartition + named-barrier sync between warpgroups, which an occupancy-first arch with static per-wave allocation lacks. **Capability-gate, do not port** (`../tile-programming/mental-model.md ## Porting a technique across architectures`). The online-softmax max/sum recurrence also blocks the simplest symmetric ping-pong (two warps cannot run identical-phase-offset softmax). |
| `gfx942` | generic `gl.load` / `gl.store` | `supported` | `local-confirmed` | First memory anchor before buffer ops. |
| `gfx942` | `cdna3.buffer_load` / `buffer_store` | `source-proven` | `local-confirmed` | Useful for streaming; element offsets (NOT byte — the emitter scales them) + typed fallback required. |
| `gfx942` | `cdna4.async_copy.buffer_load_to_shared` + `commit_group` / `wait_group` + `load_shared_relaxed` | `supported` | `local-confirmed` | **Async direct-to-LDS works on gfx942 via the `cdna4` namespace** (the `cdna3` namespace has no async submodule; generic `ttg.async_copy_global_to_local` is illegal here). Width is 32-bit (`## Direct-to-LDS granularity (per arch)`), so it often turns `s_waitcnt`-bound — A/B vs sync staging. Transpose/dot-operand offset layouts under v3+`SwizzledSharedLayout` may fail LLVM translation → use `PaddedSharedLayout`+`DistributedLinearLayout`. `compute_efficient_padded_shared_layout` is v4-only. |
| `gfx942` + Triton 3.7.0 | `cdna3.sched_barrier` / `sched_group_barrier` | `version/API-blocker` | `local-confirmed` | Same status and **same workaround as the gfx950 row**: the cluster markers emit `s_setprio` + `sched_barrier`, which is the reason a kernel on this target rarely needs the bare symbol. |
| `gfx1201` (RDNA4) | `GLOBAL_LOAD_TR_*` transpose GMEM load | `source-proven` | `upstream-doc-proven` | RDNA4 transposed global load (`rdna-fork.md`); the RDNA analog of CDNA `ds_read_tr`. Compile-verify the Gluon/asm path per build. |
| `gfx1201` (RDNA4) | async copy → LDS / TDM | `—` | `upstream-doc-proven` | **No async matrix DMA on RDNA4 ISA** (TDM is gfx1250/CDNA5/MI450 only). Use hand LDS ping-pong; `buffer_load_to_shared` async is CDNA-only. |
| `gfx1201` (RDNA4) | software prefetch (`s_prefetch_data`) | `unknown-needs-probe` | `unknown-needs-probe` | RDNA4 ISA addition; lever value + Gluon/compiler exposure unverified — probe. |
| `gfx1201` (RDNA4) | fine-grained waitcnt (`loadcnt` / `dscnt` / …) | `unknown-needs-probe` | `unknown-needs-probe` | Layer-4 pipeline lever; verify exposure + effect per build. |
| `gfx1201` (RDNA4) | dynamic VGPR (`S_ALLOC_VGPR`, wave32) | `arch-gated` | `upstream-doc-proven` | **Not an RDNA4 feature.** `S_ALLOC_VGPR` is rejected as `invalid instruction` by ROCm 7.1 / LLVM 20 on gfx1200 **and** gfx1201, with no `+dynamic-vgpr` subtarget feature — it belongs to gfx1250/CDNA5. Plan occupancy on the static ≤256 VGPR/wave model (`rdna-fork.md` §Dynamic VGPR). |
| `gfx1201` (RDNA4) | producer/consumer warp-specialization | `arch-gated` | `upstream-doc-proven` | RDNA4 has **no named-barrier warpgroups** — same capability gate as gfx950; dynamic VGPR alone does not lift it. Do not port (`rdna-fork.md`). |

## Instruction shape rules

| Target / version | `instr_shape` rule | Notes |
| --- | --- | --- |
| Triton `< 3.6` CDNA | older `AMDMFMALayout` examples may use 2D `[M, N]` | Do not copy into 3.6+ without checking local `triton_version.py`. |
| Triton `>= 3.6` CDNA | use 3D `[M, N, K]` | 2D forms like `[32, 32]` are misleading for this version. |
| `gfx950` regular MFMA INT8 | `[32, 32, 16]` or source-proven equivalent | int32 accumulator. |
| `gfx950` regular MFMA BF16/FP16 | `[32, 32, 16]` or source-proven equivalent | fp32 accumulator; verify K width locally. |
| `gfx950` regular MFMA FP8/BF8 | `[16, 16, 32]` or `[32, 32, 16]` | The only two `(M, N, K)` the `v3.8.0` MFMA v4 table registers for a non-scaled fp8/bf8 op. Any other K is a scaled-path or upcast question, not a dtype blocker. |
| `gfx950` CDNA4 scaled FP8/FP4 | `[16, 16, 128]` or `[32, 32, 64]` | The two `mfma_scale_f32_*_f8f6f4` entries; every f8f6f4 type keys onto them. |
| `gfx950` CDNA4 scaled paths | target/version-specific | Do not copy gfx942 blockers; verify local CDNA4 source, scale layout, correctness. |
| `gfx950` plain Triton matrix hints | `matrix_instr_nonkdim` is version-sensitive | Re-search on installed Triton; tuned winners can flip across versions. |
| `gfx942` BF16/FP16 | `[32, 32, 8]` or `[16, 16, 16]` | Must match local source, K width, layout contract. |
| `gfx942` INT8 | `[32, 32, 16]` or `[16, 16, 32]` | Verify accumulator + store dtype. |
| `gfx942` FP8 Gluon MFMA | blocked | Use plain `tl.dot` comparator. |

## Tile / shape constraints (Gluon language)

| Constraint | Status | Notes |
| --- | --- | --- |
| tensor shape elements (incl. `gl.arange`, tile / head dims) must be a **power of 2** | `version/API-blocker` | `gl.arange(0, 192)` -> "Shape element must be a power of 2". A non-pow2 head dim / block must be **padded to the next pow2** (e.g. `D=192 -> 256`, wasting that fraction of QK/PV flops) or **split** into pow2 pieces (`128 + 64`). This is why vendor asm ships specialized non-pow2-D kernels; in Gluon it is not a simple `HEAD_DIM` swap. A larger pow2 head dim that *is* expressible (e.g. `D=256`) can still be a clean new capability where no asm exists. |

## Deprecated / version-sensitive knobs

| Target / stack | Knob | Status | Recommendation |
| --- | --- | --- | --- |
| Gluon path, Triton 3.8.0 (both arches) | `num_stages` | **dead** — no Gluon pass consumes it | Budget parameter / champion record only; never carry plain's value over as a Gluon tuning knob. The overlap is authored (`../tile-programming/pipeline.md`). |
| `gfx950` Triton 3.7.0 | `kpack` | deprecated, overwritten to `1` | Remove from new configs; warning noise is not perf evidence. |
| `gfx950` Triton 3.7.0 | `matrix_instr_nonkdim` | active but version-sensitive | Preserve when a tuned config depends on it; re-sweep across versions. |
| Triton 3.7.0 | `@triton.autotune(use_cuda_graph=True)` | deprecation warning | New kernels should not rely on it as a stable API. |

## Build switch permission

When a task targets a build-sensitive performance class, first record whether the
user allows changing the installed Triton build. Some scheduler / register
allocator / post-assembly / pipeline-stage features are only in specific
Triton/ROCm builds; absence is a scoped build ceiling, not proof the source-level
mechanism is invalid.

```text
build switch allowed: yes | no | ask
llvm co-tuning sanctioned in prompt: yes | no
current build identity (triton, rocm, package, install policy):
required feature(s):
mechanism's validity if feature is added (best estimate):
fallback direction if not switching build:
```

If switching is not allowed, treat the missing feature as a scoped ceiling,
record the scoped blocker, and switch direction.

**Default in this skill (kernel-only, fixed build):** the build is fixed and
read-only, and **modifying compiler passes is out of scope** — a missing or
insufficient pass is a scoped ceiling +
compiler-change handoff, not a fix
(`../tile-programming/compiler-contract.md ## Compiler scope`).
`build switch allowed` is only for *switching* to another **prebuilt** build when
the user explicitly allows it, never for editing or rebuilding compiler source.

**Exception — sanctioned LLVM co-tuning:** if the user sanctions LLVM/compiler
joint tuning in the prompt, authoring a compiler pass becomes in-scope under the
Scenario B safety loop — new feature branch on the installed Triton source,
default-off `cl::opt`, single-`.a` swap + relink, `MF.verify` + A/B + restore
(`../tile-programming/compiler-contract.md ## Scenario B: sanctioned compiler
co-design`).

## Reading `hw_constants.json` across generations

Every number above is per-arch, and `hw_constants.json` (GEAK `perf_knowledge/hardware/data/`,
read via `kernel_workflow/scripts/kernel_tools/_hwdata.py`) is the SSOT for it — pass `--arch`
everywhere rather than carrying a figure across a generation (examples use `--arch gfx950`; a tool
that needs an arch and was not given one refuses rather than defaulting). Three ways that goes wrong, all of
which produce a *confidently wrong* verdict rather than a visibly missing one.

**A key absent on one arch is not permission to use the other arch's value.** Three gfx950 keys are
absent where gfx942 has them, and the first two are traps rather than gaps:

| key | why the fallback is worse than declining |
| --- | --- |
| `fp8_dtype` | gfx942 is `float8_e4m3fnuz` (**FNUZ**); CDNA4 uses the **OCP** spelling, and FNUZ is precisely the one fp8 CDNA4 does **not** have. Triton silently upcasts it to fp16 with only a warning — so the kernel runs, the numbers are "right", and it is no longer an fp8 kernel. A selector claiming only the FNUZ spelling also never matches a CDNA4 fp8 bottleneck |
| `lds_min_alloc_bytes` | this is what tells a reported `lds/WG` apart from an allocator round-up. gfx950 has `lds_align_bytes: 1280`, which **is not that granularity despite looking like it** — a measured allocation need not be a multiple of it |
| `cvt_off_f32_i4` | a binding-availability record rather than a silicon fact (see its note in the JSON); it does not bear on transcription either way |

The key that is read on the hot path, `lds_per_cu_kib`, is present and correct on both.

**A larger ceiling turns a hard failure into a silent slowdown.** CDNA4 has 2.5× the LDS/CU of
CDNA3, so a configuration that dies on gfx942 with `OutOfResources: Required <n>, Hardware limit
65536` can compile cleanly on gfx950 and be slower by a multiple. The exception disappearing is not
the problem disappearing. Before reusing a gfx942 diagnostic on gfx950, ask whether it was keyed on
*something throwing* — and if it was, find the signal that survives (register `spill=`, the pass
list, `probe.py measure`) rather than reading a clean compile as a pass.

**Several rows above change a verdict, not a magnitude.** Two worth stating outright:

- `ds_read_b128_2way_stride_bytes` exists only on CDNA4, so a swizzle that was conflict-free on
  CDNA3 is **not known to still be** — and the sign can invert, with a `per_phase` choice that
  costs a conflict on one generation being free on the other. Re-derive from the keys; do not carry
  the rule of thumb.
- `matrix_layout_family` is `version=3` vs `version=4`, so layout digests **should** differ across
  generations. Assert four-version consistency *within* one arch plus a passing round-trip. A
  cross-gen digest equality check is asserting the wrong thing and will fail on a correct port.

## Minimum probe before trusting a cell

```text
target / Triton / ROCm / PyTorch:
import path / operator path / API or op path:
dtype path / layout version / instr_shape:
compile status / correctness status / benchmark boundary:
evidence class:
```

If any field is unknown, keep the matrix entry `unknown-needs-probe`.

An entry marked `unavailable` / `arch-gated` / a feature blocker should cite a
**compile-verified probe** (a minimal kernel that exercises the feature on the
target arch, with the actual pass/error), not only a source-grep line. Source and
compile usually agree, but build flags, fallbacks, and version skew can make a
source gate misleading — verify the gate by compiling, then record the evidence
class as `compile-verified`.
