# Gluon API (gfx950 / gfx942) — mechanism index

Owned, self-contained Gluon API reference for tile-programming-gluon. **Start at
`../hardware/atlas.md`** (phase read order + backbone layer map). Organized by the
**tile-programming mechanism / backbone layer** you are working on, not as an
alphabetical API dump. gfx950 (CDNA4) is the default; gfx942 (CDNA3) is the
downgrade. RDNA4 client WMMA ISA (gfx1201; GEAK product calibration is
R9700-only): `rdna-wmma-reference.md`.
gfx1250 (CDNA5 / MI450) WMMA+TDM: structured sub-target, separate data-center fork.

Planning constants: `../hardware/planning-constants.md`. ISA shapes: `../hardware/isa-mechanisms.md`.
Support level and evidence: `../hardware/capability-matrix.md` (not restated here).
Compiler env knobs: `../tile-programming/compiler-contract.md`.

**Version baseline.** Every mechanism here is written in its **upstream Triton 3.8.0** spelling;
3.6 / 3.7 differences appear only as downgrade notes on the mechanism they affect, and the
four-version probe table is `pipeline-reference.md`. One consequence worth stating at the door:
**`num_stages` is dead on the Gluon path in 3.8.0** — no pass consumes it — so it is a budget
parameter and a champion-record field, never a knob to carry over from plain.

## Mechanism -> API route (by backbone layer)

| Backbone layer / need | Read | capability-matrix cells to check |
| --- | --- | --- |
| **which kind of kernel this is, before any mechanism** | `../workloads/index.md`, keyed by the archetype `kernel_workflow/scripts/kernel_tools/hw_budget.py --sku MI355X --workload` already requires (the pack's `scripts/hw_budget.py` is a shim to it) — it prints the row next to the ceiling. The archetype pages this pack owns sit beside it (`../workloads/{moe,collective,linear-attention,prologue}.md`); their gates are in that router's rows, not restated here | none — the archetype routes reading; the BOUND ranks levers (`../hardware/bound-class-signals.md`) |
| imports / `@gluon.jit` / launcher / host layout factories | `imports-and-launching.md` | Gluon capability level |
| memory path (`gl.load/store`, buffer ops, async copy) | `memory-reference.md` | memory & scheduling matrix |
| LDS layout (padding / swizzle for conflict-free `ds_read`) | `layout-reference.md ## Shared Layouts: The Three Constructors` for the constructors and the compile-time `gl.bank_conflicts` check, `## Let 3.8 Compute The Padded Layout For You (CDNA4)` when the consumer is an MFMA operand — that section also carries the per-buffer selection rule (this factory **iff** the buffer is read back as a matrix operand, `SwizzledSharedLayout` otherwise), and `memory-reference.md ## Deriving the producer layout for a padded async destination` is its other half when the buffer is filled by async copy — then `../tile-programming/layout-recipes.md` for the recipe | `Swizzled`/`PaddedSharedLayout` rows |
| runtime-indexed access (lookup table, routing map, paged index chain) | `layout-reference.md` (register-level, layout-conditional) + `smem-lds-reference.md` (LDS-level) | none — both are layout/LDS decisions, not capability cells |
| the indexed access has to be a **read-modify-write** (LDS histogram, CTA-local counting sort, slot reservation) | `smem-lds-reference.md ## Atomic RMW in LDS — the atomic_scatter_* family` — and read the naming note there first: the only spelling is `atomic_scatter_<op>`, so searching for an atomic-add name on a descriptor returns nothing whether or not the mechanism exists | LDS atomics on a shared descriptor are **3.8.0 only**; `.gather` / `.scatter` are 3.7.0+ |
| matrix (MFMA result/operand/convert/acc, scaled MFMA) | `matrix-reference.md` | matrix-op matrix + instr_shape rules |
| pipeline / slicing / scheduling — **including the order to reach for pipeline mechanisms** (hand-written first: register prefetch, authored LDS ring, `warp_pipeline_stage` + scheduling model; re-injection last, as diagnostic / last resort) | `../tile-programming/{pipeline,slicing,compiler-contract}.md` — the order is defined there, once | version-sensitive knobs |
| the joint VGPR+AGPR budget will not fit, or occupancy dropped a step and you cannot find the work to remove | `../tile-programming/slicing.md` owns the slicing layer and is where you start. When it is exhausted, there is one more lever it does not cover: a value living longer than it needs to, which the allocator pays for in accumulation registers, in scratch, or in an occupancy step. Its four entry numbers come out of the static `.amdgcn` with no GPU — `inline-asm/deciding.md ## Lane 0 — structural facts, readable from the kernel alone`, rule **L0.7**, which stays locked until **L0.0** has found no headroom left to spend. The site that results emits no instruction, so no profile will ever point at it | none — this is an allocator outcome, not a capability cell |
| the overlap exists but the instructions clump — pacing it (scheduling fence, `s_nop`, `llvm_fn_attrs`) | `../tile-programming/instruction-scheduling.md` for where in the stream they land; the first two are inline asm, so `inline-asm-reference.md ## Class 2 — scheduling control` owns what the block can and cannot do — notably that the empty-asm form is a compiler motion boundary and not a memory fence. `llvm_fn_attrs` is the odd one out and has its own chapter, `../tile-programming/llvm-fn-attrs.md`: it is a JIT compile option shared by the plain and Gluon lowerings, so **nothing about it is Gluon-specific**, and it is gated on the kernel being ILP-starved | `llvm_fn_attrs` is **3.8.0 only**; `sched_group_barrier` / `iglp_opt` have **no surface on any version** |
| which scheduling model, then the wave-level one | `../tile-programming/scheduling-model.md`, then `../tile-programming/warp-pipeline.md` | `warp_pipeline_stage` is **absent on 3.6** |
| `warp_pipeline_stage` / async-copy API surface and the four-version probe table | `pipeline-reference.md` (API router only — it does not rank) | the authority for what this build has |
| low-precision side path (FP8/FP4 scaled) | `matrix-reference.md` + `../tile-programming/low-precision.md` | FP8/FP4/scaled rows |
| AOT / prebuilt / wrapper packaging contract (triple, launch attributes, signature, fallback selection) | `shared-aot-reference.md` — shared memory and async copy are **not** here; it routes to their owners | AOT rows |
| gfx950 smoke / minimal probe | `gfx950-minimal-examples.md` | Gluon capability level |
| **you are about to WRITE inline asm, or to record a language ceiling** — three separate jobs wear this syntax: the language has no spelling for what the ISA does (a wave-collective, a register-file pin); an ordering or a machine-state bit is part of **correctness**; or the site computes nothing at all and exists only to constrain the allocator and the scheduler. Which one you hold changes the removal test, the acceptance test, and whether a measurement is even admissible | `inline-asm-reference.md ## Three jobs, one syntax` first, to decide which of the three you are in, then `## Before you reach for it: what the language already emits` — most reasons to want it are already expressed — then `## Classifying a site: mechanism × intent` to bin what you are about to write | read the `Absent` cell **and** this page before recording a language ceiling; a ceiling the chapter answers is not a ceiling |
| **you are reading, auditing or lifting an asm site someone else wrote** | `inline-asm/field-guide.md ## Reverse index: I see X in an unknown kernel`, then `## Traps, ranked by how likely you are to hit one`. If the site is half a pair or keyed on a tile constant, neither reads correctly alone — `## Shape-keying: inline asm is a per-M specialization` | none — this is a reading task, not a capability question |
| which ISA atom a mechanism is actually made of — the MFMA / scaled-MFMA / `ds_read_tr` / copy atoms per target, and the RDNA and gfx1250 downgrades | `atoms-reference.md` | read it with the matrix and memory cells; it names the instruction, the matrix names the support |
| "does this API exist / what is this one-off name" — single-point APIs too small for a mechanism section, and the spellings that **do not exist** | `appendix-api.md` | none — existence and signatures are source-proven against the four versions there; support claims still live in the matrix |
| you arrive with a finished kernel behind you: which of its conclusions carry to the next one unchanged, which need a quantity recomputed, and which **reverse sign** | `technique-transfer.md` | none — every cell routes to the owner page that already establishes it |
| compile / lowering / correctness failure | `../method/triage.md`, then `../pitfalls/platform-known-issues.md` for target- and version-sensitive lowering traps | the failing target/dtype cell |
| a Gluon route looks plausible and may be the wrong layer (admission, Quick Reject, measured negatives) | `../pitfalls/negative-patterns.md` | the `wrong-result` / blocker cells |

## Lowering-chain order (matrix path)

result layout -> operand layouts -> `convert_layout` -> accumulator -> target op
-> epilogue/store layout. Full chain in `matrix-reference.md`.

## gfx950 <-> gfx942 at a glance

gfx950 (CDNA4, MI350X / MI355X) is the main line; every page of this reference writes the gfx950
form first. gfx942 (CDNA3, MI300X / MI325X) is the downgrade. Numbers come from the per-arch keys in
`perf_knowledge/hardware/data/hw_constants.json` (pass `--arch` to every tool; none defaults one).
Each row names the page that owns the fact — this table only routes.

| Aspect | gfx950 (CDNA4, default) | gfx942 (CDNA3, downgrade) | owner |
| --- | --- | --- | --- |
| MFMA op / layout | `gl.amd.cdna4.mfma`, `AMDMFMALayout(version=4)` | `gl.amd.cdna3.mfma`, `version=3` | `matrix-reference.md` |
| scaled MFMA | `cdna4.mfma_scaled` (required for FP4 at any shape, and for FP8 at an `(M, N, K)` the regular table does not register) | **none** — cannot-select; plain `tl.dot` comparator | `atoms-reference.md` |
| FP8 / FP4 | native OCP `e4m3fn` / `e2m1`; op selection is per `(version, M, N, K)`, not per dtype — regular `cdna4.mfma` covers fp8 at `[16, 16, 32]` / `[32, 32, 16]`, `mfma_scaled` (None scale -> unit e8m0) covers FP4 and the unregistered fp8 shapes | **FNUZ** fp8 (`e4m3fnuz`, upcast with a warning if it reaches gfx950); `cdna3.mfma` OCP FP8 version/API-blocker; plain `tl.dot` (scale-less `tt.dot_scaled`) ok as comparator | `matrix-reference.md ## Matrix-Family Details` |
| async global→LDS | `cdna4.async_copy` + `commit_group` / `wait_group` ring at **128 or 32 bits** per thread | the same entry points at **32 bits only**, clean tiling, destination `order=[1, 0]`; measured slower than **sync staging**, which is the default there | `pipeline/async-ordering.md`; `pipeline/marker-and-version-gates.md ## The gfx942 async-copy width gate — available, and narrow` |
| transpose LDS read | `ds_read_*_tr_*` emitted by the backend (no source API) | **absent** | `atoms-reference.md` (its `ds_read_tr` (gfx950) section) |
| LDS per CU / banks | **160 KiB / 64 banks** | **64 KiB / 32 banks** | `smem-lds-reference.md` |
| compiler-derived padded layout | `cdna4.compute_efficient_padded_shared_layout` (3.8.0) | none — hand-computed | `layout-reference.md` |
| buffer ops | `gl.amd.cdna4.buffer_load/store` | `gl.amd.cdna3.buffer_load/store` | `memory-reference.md` |

Namespace is not architecture support; full status + evidence live in
`../hardware/capability-matrix.md`.

## The TTGIR -> Gluon bridge

When transcribing a plain-Triton kernel, the layouts are recovered by
`scripts/ttgir_bridge.py`, which hands the champion's `.ttgir` to the compiler's own MLIR parser
and then to `layoutToGluon()` — upstream's own attribute→`gluon.language` converter. There is **no
mapping table** in this pack to keep in sync with Triton; equivalence is decided on LinearLayout
normal forms, not on attribute text. A TTGIR layout kind with no `gluon.language` constructor on the
build surfaces as a named **`UNRECOVERABLE`** row (the one seen in practice is
`amd_rotating_shared`); do not substitute a similar layout. Probe the build, re-dump at `ns=1` and
re-recover; if it is still unrecoverable, record a forced divergence in the ledger or a
`structure_suspect` with the kind named. The procedure is `../method/transcribe.md`; the
padding / swizzle recipes the recovered layouts feed are `../tile-programming/layout-recipes.md`.

## Wave-size note

gfx950 and gfx942 are both **wave64** (`product(threads_per_warp) == 64`). All
layout recipes here assume wave64.
