# Gluon Negative Patterns

When a Gluon route is plausible but may be the wrong layer, or an edit "should" help and did not.
gfx950 (CDNA4) / Triton 3.8.0 is the main line; gfx942 entries say so. Per-target support:
`../hardware/capability-matrix.md`. Platform traps (lowering, toolchain pins, process aborts,
profiler gaps): `platform-known-issues.md`.

**Suggested: encode validated values as code, not prose.** A rejected config gets re-proposed even with the
note open; a `gl.static_assert` (number + artefact path in the message) catches it at compile time.
Where no formula exists (e.g. `k_width` at an `mfma_scaled` site) pin the numerically validated
value and assert on it. Assert a shipped default on the *reporting* path, not in the A/B harness
(the harness sets the value it varies, so it cannot see a wrong default).

## Quick Reject Checklist (= gate skip-transcription conditions)

Signals that the plain-Triton tier (`../method/front-end.md`) is likely the better layer; the more of them
hold, the stronger the signal:

- the hot path spans several `@triton.jit` helpers and tensors cross the JIT boundary, or one idea
  needs the whole call chain rewritten as `@gluon.jit`; partial migration through `tl.dot` (its
  result carries no Gluon layout);
- several dots with different tile shapes, or dot + reduction + atomic constraints in one patch;
  operands/online state/masks/stores all need different parent layouts (migration radius > gain);
- plain Triton already lowers the hot dot to the desired MFMA and the remaining bottleneck is
  config, wrapper, dispatch or memory traffic; or the Gluon version is the same algorithm with no
  explicit layout / memory-path / scheduling / matrix mechanism;
- `../hardware/capability-matrix.md` marks the target/dtype/op `wrong-result` or `version/API-blocker`;
- the kernel is launch/wrapper dominated;
- decode-shaped or paged K/V — an **admission hint**, read `## Paged / Indirect Decode` first;
- **`#ttg.amd_rotating_shared` in the plain TTGIR has no `gluon.language` constructor.**
  `scripts/ttgir_bridge.py` reports it as `UNRECOVERABLE` (`ttgir_to_gluon.py` emits
  `None  # NOT EMITTED: amd_rotating_shared ...`). It is a linear layout, so a `SharedLinearLayout`
  expresses it exactly: take the bases from upstream `sharedToLinearLayoutAMDRotating`
  (`LinearLayoutConversions.cpp`, 3.8.0): column bases `[0, 2^i]`, then for each row power `r`,
  `[r, (vec * (((r // perPhase) % maxPhase) ^ ((r // (perPhase * maxPhase)) % maxPhase))) % numCols]`
  (order `[1, 0]`). Derive them with that formula, not by eye, and check the result element by element
  against the formula (bijective and matching at `vec=4, perPhase=4, maxPhase=4`, `[128, 64]`; compiles on
  gfx950 as `#ttg.shared_linear`; kernel-level speed of the substitute not measured). Re-dumping at
  `num_stages=1` may also remove the layout (`../method/transcribe.md ## 2. Recover`).

## No-Extra-Mechanism Decision

If plain already lowers the hot path, Gluon removes no traffic / conversion / launch, the Gluon
phase is a small share of the boundary, and the expected gain is under noise, a Gluon port has no
mechanism to gain from (`no_extra_gluon_mechanism`); a newly identified mechanism re-opens it.

## Paged / Indirect Decode

Gluon helps a page-table / gather chain (`page_load -> kv_loc -> data_load`) only if it cuts bytes,
changes the address chain, improves locality or removes a measured conversion.
**L2-resident gathered K/V is load-latency / MLP-bound, not HBM-bound**: signal `MemUnitStalled ~ 0`
with the busy counter well under 100% (`../method/profile.md ## Derived metrics`). Wider loads and
HBM levers are not expected to move it (the bound is latency, not bytes); occupancy / memory-level
parallelism is the matching lever family, and software prefetch that costs waves can regress when
VGPR/LDS already cap them — A/B before closing either (`../tile-programming/pipeline.md ## Budget before deepening`).
**Per-token async direct-to-LDS gather lowers when each lane's chunk is contiguous and aligned**:
`buffer_load_to_shared` with offsets `tok[:, None] * stride + col` (stride constexpr, or JIT-specialized as a
multiple of 16) lowers to `buffer_load_dwordx4 ... lds` with no `ds_write` on gfx950 (32-bit
`buffer_load_dword ... lds` on gfx942; compile-checked 3.8.0). Per-lane global addresses may be arbitrary;
only the LDS-side write must be coalesced. Gluon runs no `CoalesceAsyncCopy`, so when alignment cannot be
proven (unspecialized runtime stride) lowering fails with `failed to translate module to LLVM IR`; there is
no fallback to register staging. A sparse-MLA random-row gather has run as a double-buffered async DMA.

## Fused / Block-Scaled Matrix Kernels

Weak rewrite candidate when one body mixes matrix paths with different dtype/scale/instr-shape
contracts, the source is `dot(a,b)*scale` (scale-before-add, which an unscaled MFMA
accumulator cannot express), or scale granularity mismatches the instruction's (hot-loop scale replication). Build a
phase map (dtype, scale, ideal tile, accumulator lifetime per phase) before editing.

## Bandwidth-Ceiling Refinement

When bandwidth is near the practical ceiling, an edit that cuts instructions rather than bytes has
little left to gain. **Exception: spills are hidden memory traffic** — lowering `waves_per_eu` or changing the
config can cut HBM bytes with no visible load/store change; check register pressure first.

## Hot-Loop Layout Conversion

A `convert_layout` inside the inner loop is paid every iteration, so its cost scales with
`K / BLOCK_K`; the per-iteration count is a cheap signal to read alongside launch knobs. What changes
the picture is a hoist point (host layout contract, phase boundary, matrix-operand boundary,
consumer-layout load, shared staging) or a matrix/memory mechanism that removes more than the
conversions add.

## `warps_per_cta` is not an independently tunable knob

**Changing only `mfma_layout.warps_per_cta` crashes with an unattributed LLVM assert:**

```
llvm/ADT/Sequence.h:275: iota_range::iota_range(T, T, bool):
  Assertion `Begin <= End && "Begin must be less or equal to End."' failed.
```

GPU-measured on gfx950 (triton 3.7.0+amd.rocm7.2.0, MLA forward, `[4, 1] -> [2, 2]`): the kernel pins
**14 hardcoded `warp_bases` tuples** across blocked, linear and dot-operand layouts (one the
degenerate `warp_bases=((0, 0), (0, 0))`), all written for 4-warps-along-M. The crash is that
incoherence, not an arch/build limit. Fix: `grep -c warp_bases` to size the change; treat it as a
coupled multi-round layout rewrite (plus the cross-warp exchange the new split needs). The prize
is real (`[N, 1]` broadcasts B to all N warps, reading K/V N times from LDS) but not a one-liner.
On 3.8.0 (compile-only, gfx950) the same kind of change on a small kernel with literal warp bases
failed loudly instead (`Result has an invalid layout` at the non-native `num_warps`); the assert above is
the observation on the 3.7.0 ROCm build.
**Same for sweeping `num_warps` on a transcribed kernel**: the bridge emits the IR's literal
`warps_per_cta`, so another warp count is a compile failure (`Result has an invalid layout` in the cases
checked) or a correctness bug, not a slow config
(`../gluon/layout-reference.md ## BlockedLayout Constraints (wave64)`). Autotune tiles/stages with
`num_warps` pinned; a tile retune re-recovers every layout (plain front end).

**Before calling a sweep exhausted, list every quantity the kernel pins as a literal** (warp map,
per-thread width, stage count, shared-layout params) and mark each swept or fixed-with-reason — a
grid cannot visit an axis it has no name for.

## A translation failure names the allocation, not the murderer

**`builtin.unrealized_conversion_cast` at LLVM translation prints the location of the shared
allocation, not of the op that has no lowering.** Bisecting by deleting ops around the allocation one
at a time until the message disappears localizes it; swapping layouts moves the combination and can
hide the cause. Known causes on `buffer_load_to_shared`: per-thread width
outside the arch set (the most frequent one seen —
`../gluon/memory-reference.md ### Minimum per-thread granularity (applicability by dtype)`);
padded destination whose producer is not derived from its `.offset_bases` or whose interval is shorter than one wave's copy
(`../gluon/memory-reference.md ## Deriving the producer layout for a padded async destination`).
A third, unsuspected op is always possible. It is a `(layout, shape, dtype, op)` coverage gap, not a
build ceiling. Verified on Triton 3.8.0 / gfx950.

## Instruction count is not the objective function

**At low occupancy, VALU that co-issues with MFMA can be hidden behind it; VALU on the dependent
dot -> reduce chain or after the last MFMA tends to be exposed.** A census cannot tell them apart. GPU-measured, gfx950
MLA forward, register- and LDS-bound at 1 wave/SIMD, interleaved A/B:

| edit | census | result |
| --- | --- | --- |
| transpose V on read instead of in registers | **−51** instr, mfma/ds_read identical | **slower** |
| hoist V `local_load` above the softmax | reorder only | **slower** (full drains 4→6, `s_nop` 7→11) |
| skip `acc *= re_scale` behind an exponent-margin test | **−64** `v_pk_mul_f32`/iter | **slower, the largest loss of the four** (branch + reduce cost more) |
| fuse three epilogue `acc` scales | −2 tile multiplies | tie (within spread: not resolved) |

- Locate before deleting: which side of the last MFMA, and is anything there to co-issue with?
  Occupancy from `kernel_workflow/scripts/kernel_tools/probe.py measure --dir <ir_dir> --arch gfx950`.
- A pipe at parity is not a pipe to feed: the MFMA and LDS issue models were each well under the measured
  cycles per iteration; giving either more slack bought nothing (twice). Compare each pipe's modelled
  cycles/iter (`amd_isa.py facts`, `asm_loop_audit.py`) with the measured loop time before feeding it.
- Report count and time separately; a removed pattern is a falsifier, not a price
  (`../method/benchmark-hygiene.md ## A compile-only kill step is a falsifier, not a price`,
  `## Before timing, list every quantity the edit moves`).
- **A width census bounds issue slots, not bytes, when the redundancy is across lanes** (adjacent
  lanes coalesce;
  `../tile-programming/memory-path.md ## A load's width is issue-side; a transaction's width is access-side`).
- **`BlockedLayout` over-coverage fetches the same byte twice in one lane.** If
  `size_per_thread * threads_per_warp * warps_per_cta` on a dim exceeds the tile extent (measured
  1·4·4 = 16 over 8), every byte of that tensor is loaded twice per lane — coalescing cannot merge
  it (same lane) and the trace instruments tried did not show it; the layout shows it. Cutting coverage to the tile (fewer bytes/lane and sub-dword loads) was faster in
  every bucket measured on that kernel (GPU-measured, gfx950). The duplicate loads mostly hit in cache, so the cost varies: a streaming microbenchmark
  (gfx950, 3.8.0) tied at twofold over-coverage and lost clearly at fourfold on a large stream — cut
  coverage to the tile and A/B. The layout legality rule permits masked over-coverage, so it passes
  silently. Check broadcast operands
  on the non-reduced dim too, and report sub-dword count **and** bytes/lane per tensor (one scale-tile
  widening was slower on one operand and faster on the other — A/B each tensor separately).

## Conditional rescale skip — measured negative at the occupancy its own path A asks for

**The losing form added a cross-lane vote.** A per-row skip pays only as a real branch: one
`gl.map_elementwise` per lane covering all of that lane's accumulator elements
(`pack = acc.numel // (num_warps * 64)`; multiply only if any alpha != 1) lowers to `s_and_saveexec` +
`s_cbranch_execz`, which skips the multiplies whenever no lane of the wave needs them (effectively
wave-uniform at runtime). A per-element `if` (`pack=1`) is if-converted by LLVM to `v_cndmask`, the same as
`gl.where`, and saves nothing (compile-checked gfx950 / 3.8.0). The measured loss came from a different form: a compare,
a cross-lane reduce for a CTA-wide predicate and a second softmax path, on a gfx950 MLA forward at
1 wave/SIMD (register- and LDS-limited): **slower** (row 3 of the table above). The vote form measured
negative in that kernel; the `gl.where` select form lost on a separate 2-wave prefill microkernel
(gfx950 / 3.8.0). The branch form is the one to A/B, and
`s_cbranch_execz` around the multiply block in the `.s` shows it landed. The FA4-style lever description
is in `../workloads/flash-attention-structural-insights.md`.

## Give a ratio a mechanism ceiling before you believe it

**Compute the most the claimed mechanism can produce before accepting a ratio.** Example: fp32 vs
bf16 atomic epilogue reported a ratio above 2x; bf16 lowers to a packed atomic (2 elements/instr) and
fp32 to 1, so the ceiling is 2.0x — the arms differed in more than dtype. An isolated re-measure was far
smaller and differed by band. Over the ceiling => suspect the arm, not the hardware. Also: a ratio
measured in one shape band is not a property of the op (atomic vs plain store differed by more than
an order of magnitude between decode and the top of prefill M, `../workloads/moe.md`); carry the lever ranking and mechanism, keep the
number attached to its band, and read per-point data before an aggregate.

## Inline asm: justify the reach before taking it

Gate for **adding** `inline_asm_elementwise`. The block is opaque to the optimizer in its region, and
the commonest form (empty body, constraint-only scheduling) is the least durable across compiler
upgrades and fails silently (`../gluon/inline-asm-reference.md ## What it costs you, on every class`).
Answer before the edit: (1) what the language already emits
(`../gluon/inline-asm-reference.md ## Before you reach for it: what the language already emits`);
(2) what breaks if deleted (`## Classifying a site: mechanism × intent` there); (3) which
disassembly check shows it landed; (4) the precondition, written next to the kernel.
The language is usually the better fit when: the construct already exists; the kernel is a pure elementwise
epilogue; the reason is "production had one" or "it will be faster"; or **the need is a branch over
Gluon-level code — asm takes and returns tensors, it cannot enclose a region to skip.**

## Removing synchronization: a green run is not the evidence you need

**A value gate (bit-identical repeats, oracle tolerance, race-test) is zero evidence that a removed
barrier / weakened scope / dropped acquire was safe** — it samples only interleavings that occurred.
Evidence is the release/acquire pairing at its scope, or the emitted ISA
(`../gluon/inline-asm/classes-sync.md ### A gate on the value cannot see a missing acquire`;
multi-device: `../workloads/collective.md`). Otherwise record the speedup as unverified.

## Knobs and spellings that compile and then cost you

The general list (`num_stages` unconsumed on the Gluon 3.8.0 path, no-op `sched_barrier` / `set_prio` stubs, `gl.warp_specialize`
failing on CDNA, inert env vars, `waves_per_eu`) lives in `../../playbook.md ## Knobs & pitfalls`; the
GEMM-style scheduler emitting invalid IR with VALU between matmuls in
`../tile-programming/compiler-contract.md ## AGPR hint workload gate: accumulator read-cadence` and
`../hardware/optimization-gotchas.md` (row 7); re-injection traps in
`../gluon/pipeline/reinjection.md ### Conditions for the pass to fire, and for it to pay`.
Only here:

- **`disableSched`** — reported to cost occupancy outright; A/B with occupancy read before using it.
- **Async copy "does not exist" on this arch** — on gfx942 and gfx950 it exists (Triton 3.8.0):
  `platform-known-issues.md ### Async copy: available on both generations, and easy to misdiagnose as unavailable`.
- **Runtime buffer indices** — legal; unrolling them pays when the ISA shows it bought something
  (`../gluon/pipeline/authored-overlap.md`, A5 example).

## Target-Specific Matrix Blockers (scoped)

Do not tune around a capability-matrix `wrong-result` / `version/API-blocker` cell; keep it scoped
to target/dtype (a gfx942 fp8 cell is not a gfx950 one). **On gfx950, `no matching matrix core
intrinsic` for fp8 usually means one unregistered `(M, N, K)`, not a dtype prohibition**
(`../gluon/matrix-reference.md ## Matrix-Family Details`).

## gfx942 / CDNA3 negative-result signatures (mechanism + error text)

Recorded on gfx942; re-check before carrying to gfx950 (160 KiB LDS vs 64 KiB; `ds_read_*_tr_*`
replaces in-thread transpose).

1. **LDS-cap-bound large-M bf16/int8 GEMM:** explicit Gluon (even re-injected + async + ping-pong)
   trailed the tuned plain pipeline in the gfx942 cases recorded (runtime ranking not re-measured — no gfx942
   device; a compile check agrees that the 64 KiB LDS cap limits such a tile to about two stages); re-test
   when the LDS budget changes (gfx950). Label re-injected numbers `injected`.
2. **A hand register double-buffer can beat the compiler pipeline** — read VGPR/spill from the
   `.amdgcn` KD before deepening (gfx942 async only at 32 bit/thread:
   `../gluon/pipeline/marker-and-version-gates.md ## The gfx942 async-copy width gate — available, and narrow`).
3. **`asm_loop_audit.py` shows `MFMA↔VALU transitions = 0` + a long MFMA clump with the pipeline
   on** (relaxed `lgkmcnt`, `s_nop=0`): suggested reading — a scheduler ceiling rather than a kernel bug (triage heuristic, not a measured rule). The stock `coexec`
   strategy is gfx1250-only. Check whether the loop's MFMA shape is in the scheduler's cost model
   (unpriced shapes drop the region silently) and whether VALU demand fits the MFMA shadow
   (`../tile-programming/llir-codesign.md`).
4. **`TritonAMDGPUInThreadTranspose.cpp:526 PassManager failed`** on a zero-copy V-transpose layout:
   explicit transpose/staging is the recorded workaround; it is a lowering ceiling of that build.
5. cshuffle epilogue LDS process-abort and 6. scaled-MFMA cannot-select (a4w4 / a8w4 / mxfp8):
   `platform-known-issues.md ## gfx942 / CDNA3 hard failures that affect benchmark validity`.

## Negative Result Record

Record: direction, Triton + ROCm + arch, mechanism tested, correctness, measured boundary, dominant
overhead, `failure_class` (values in `../method/triage.md`), next valid direction.
