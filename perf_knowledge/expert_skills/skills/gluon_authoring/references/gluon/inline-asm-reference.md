# Gluon Inline Assembly (gfx950 / gfx942)

> **Lookup, do not recall:** hazard NOPs next to an MFMA — `python3 kernel_workflow/scripts/kernel_tools/amd_isa.py --arch gfx950 hazard` / `waits`; what `vmcnt`/`lgkmcnt` count — `... --arch gfx950 waitcnt`; an instruction's encoding — `... --arch gfx950 encoding <INSTR>`. What the instructions do: `../hardware/isa-mechanisms.md`.

## Three jobs, one syntax

`gl.inline_asm_elementwise` serves three unrelated purposes; decide which one a site is before anything
else (`inline-asm/classifying.md ## Axis I — intent`: delete it, rebind its output to its input, ask what changes).

| purpose | what the site buys | what breaks if you delete it | intent bins |
| --- | --- | --- | --- |
| **Vocabulary** — the ISA can say something the language cannot | an instruction with no Gluon spelling | same answer, different (usually worse) instructions | I-C |
| **Semantics / synchronization** | an ordering obligation, a machine-state bit, a coherence qualifier | **the answer changes**, or an intermittent race | I-A, I-B |
| **Placement** — constrains allocator and scheduler | a compiler *decision*; emits no instruction | nothing visible — same answer, different allocation/schedule | I-D (I-C when a register-class pin changes downstream instruction selection) |

- Vocabulary sites keep a plain-Gluon `else:` arm; semantics sites are verified for correctness, not A/B'd.
- An empty body (`asm=""`) emits zero instructions, so PC-attributing profilers show nothing at its
  address; its value is a property of the surrounding code (`inline-asm/misreading.md`).
- No performance number for *your* kernel is carried here; a placement site is justified only by an A/B
  on your kernel (`inline-asm/deciding.md`).

| Read this | When |
| --- | --- |
| `inline-asm/deciding.md` | deciding whether to write one (structural rules L0.x, then dump/measure) |
| `inline-asm/classifying.md` | holding a site someone else wrote (mechanism × intent, `PAIRED-WITH`, `BUDGET-GROUP`) |
| `inline-asm/shape-keying.md` | writing the call: `pack`, the arity law, `is_pure`, constraint letters, bins |
| `inline-asm/classes.md` / `classes-sync.md` / `classes-lane-register.md` | per-class rules (1–3 / 4, 4b / 5, 6) |
| `inline-asm/field-guide.md` | reading unknown asm: constraint/mnemonic reverse index, traps, idioms |
| `inline-asm/misreading.md` | a profiler reading disagrees with expectation |
| `inline-asm/costs.md` | verification loop, durability across Triton upgrades, the removal test |

**Auditing a tree for sites: parse, do not grep.** Select every `ast.Call` whose callee name *contains*
`inline_asm` (it is reachable as `gl.` and `tl.`); `ast.walk` also visits a chained `.to(...)` whose
unparsed callee contains the inner call (double count); line greps also hit multi-line imports.

## Before you reach for it: what the language already emits

| The temptation | Already expressed as | Note |
| --- | --- | --- |
| cache-scope bits on an atomic (`sc0` / `sc1`) | `gl.atomic_*(..., sem=, scope=)` | `scope="sys"` emits the full system-scope sequence incl. L2 writeback and wait |
| L2 writeback before publishing a flag | the `sem="release"` atomic that publishes it | bare `buffer_wbl2` only for a publish the compiler does **not see as a release** (not the same as "not an atomic") |
| `buffer_wbl2` / `buffer_inv` around peer-device buffers | allocate the peer buffers uncached (`hipDeviceMallocUncached`) | no cache maintenance left; only the drain before the flag (`../workloads/collective.md ## The other route: uncached peer memory, plain flag stores`) |
| `sc0 sc1` on a poll or a payload load | `gl.load(..., cache_modifier=".cv")` / `buffer_load(..., cache=".cv")` | both emit `sc0 sc1` on gfx942 and gfx950 (3.8.0 ISA); `volatile=True` emits no bits, and `.cg`/`.cs` are not system-coherent (global `nt`, buffer `sc0 nt`). Asm only for a whole poll loop in one block (`inline-asm/classes-sync.md ## Class 4b — protocol blocks`) |
| `sc0 sc1` on a word another agent polls | `gl.store(..., cache_modifier=".wt")` / `buffer_store(..., cache=".wt")` | both `sc0 sc1` on both arches; the `gl.store` form adds a trailing `s_waitcnt vmcnt(0)` per store, the buffer form does not (a store-heavy loop wants the buffer form). A `.wt` flag is **not a release**: across XCDs it can overtake plain payload stores (GPU-checked gfx950 / 3.8.0) — a cacheable payload needs a release atomic or `buffer_wbl2` before the flag |
| `s_barrier` | `gl.barrier()` | on Triton 3.8.0 / gfx950 it emits no `vmcnt` wait for outstanding global stores (compile-only check); a publish still needs the per-wave drain |
| priority toggling around an MFMA block | `warp_pipeline_stage` markers | lowers to `s_setprio` + `sched_barrier` on 3.8.0 / 3.9.0 (symbol 3.7.0+, but 3.7.1 emitted no nonzero `s_setprio`; `pipeline/marker-and-version-gates.md`) |
| waiting for outstanding async copies | `commit_group` / `wait_group` | hand-written `s_waitcnt` loses the compiler's dependency checking (`## Class 4 — synchronization`) |
| skipping work on a warp-uniform condition | `gl.warp_predicate` | vendor-fork only, absent from the 3.8.0 wheel; the elementwise asm door cannot branch over Gluon-emitted code, so it does not replace it (`classes-sync.md`) |
| ordering LDS reads against the next copy | `gl.barrier()` + `wait_group` | an empty asm with `~{memory}` is **not** this (`## Class 2 — scheduling control`) |
| draining this wave's LDS **readers** before a slot is refilled | **no binding — real gap** | `wait_group` retires the producer only (`inline-asm/classes-sync.md ### The drain marker — a different obligation from wait_group`) |

### The complement: gaps that blind rebuilds actually hit

Capabilities that rebuilds of asm-carrying kernels restricted to upstream Gluon could not reconstruct.
In the upstream rebuilds most kernels needed none of them and closed the gap by recovering headroom
(`inline-asm/deciding.md`, L0.0); the table lists what remained there, not a set of levers to adopt.

| The gap | Why the language does not reach it |
| --- | --- |
| **Split-K across warps** | the MFMA layout's warp partition divides the *output* and is pinned to `num_warps`; no spelling gives two warps the same output tile and different K slices |
| **Instruction order within a wave** | on 3.8.0 no free-standing `sched_barrier` / sched-group / `s_setprio` call in `gl.*` or `gl.amd.*` (priority only through `warp_pipeline_stage` markers); `schedule_hint` is declared but never read by the 3.8.0 AMD backend |
| **LDS-free fused MFMA epilogue store** | coalescing MFMA results into wide stores needs a lane permutation the tensor level only expresses via an LDS round trip |
| **Asserting a value is wave-uniform** | on 3.8.0 no construct pins an SSA value into an SGPR; `gl.assume` takes a value predicate, not a uniformity hint |
| **A bare per-wave drain of global stores before a barrier (publish fence)** | `gl.barrier()` emits no store wait (3.8.0 / gfx950, compile-only); a release atomic is understood to drain only its own wave (unverified at run time) and adds a writeback (`inline-asm/classes-sync.md ### The publish fence — drain, barrier, opaque zero`) |
| **Wall-clock time and trap in a spin** | no clock / trap builtin in `gl.*` (checked on the 3.8.0 wheel); an iteration bound is the API-only fallback |
| **Keeping a long-lived value out of AGPRs** | register class has no surface; past an occupancy step the allocator may sink a long live range into AGPRs (the gap the placement purpose fills) |

## The only door: gl.inline_asm_elementwise

```python
out = gl.inline_asm_elementwise(
    asm="...",              # $0.. operand placeholders
    constraints="=v,v",     # LLVM constraint string: outputs first, then inputs, then ~{clobbers}
    args=[x],               # input tensors, broadcast to a common shape
    dtype=gl.float32,       # result element type(s); a tuple gives a tuple of tensors
    is_pure=True,           # "no side effects"
    pack=1,                 # elements per asm instance (per lane)
)
```

Present on 3.6.0–3.8.0 with the same signature, re-exported from Triton core (on the 3.8.0 wheel it is the same object as
`tl.inline_asm_elementwise`, and it is the only `asm` entry point in `gl`).
**After 3.8.0 upstream added a second, Gluon-only door `gl.inline_asm` (`ttg.inline_asm`)**: one
invocation per thread (no `pack`, no arity law), results optional, explicit distributed layouts for
results, memdesc operands allowed (upstream's docstring requires `is_pure=False` for them). Not on the 3.8.0 wheel; the rules below are rules
of *this* door.

- **Elementwise only on 3.8.0.** No scalar/statement-level asm in Gluon on that wheel; machine-state and sync blocks are
  smuggled through a tensor map.
- Basic contract (≥ 1 result, dummy result for effect-only blocks, `is_pure`, <4-byte input packing,
  `$n` numbering): `triton/language/core.py` `inline_asm_elementwise` docstring. Skill-only additions below.
- **Consume an effect-only block's dummy result via a tie** into a value the following code uses — that is
  its ordering anchor; unconsumed + `is_pure=True` ⇒ deleted.
- **Which elements share an instance is unspecified; the bit order within a pack is not**: lower
  element indices occupy lower bits (stated only in an upstream issue comment; relied on by code).
  To get a lane index, pass `gl.arange` under a layout whose element order is lane order as an
  operand (`## Class 5 — cross-lane and wave-collective`).
- `is_pure`: `inline-asm/shape-keying.md ### is_pure — a decision rule, not a description`; arity:
  `inline-asm/shape-keying.md ### The arity law — pack, dtype and the constraint string are one decision`.
- **Narrow outputs pack too:** a `pack=8` bf16 output is 4 slots ⇒ first input `$4`. An argument may
  legitimately appear twice (vector + scalar class).

## Classifying a site: mechanism × intent

Two axes — **M** (read the asm and constraint strings) × **I** (delete + rebind: what changes) —
plus `PAIRED-WITH` and `BUDGET-GROUP`: `inline-asm/classifying.md`.

## Class 2 — scheduling control

Empty tied asm / `s_setprio` / standalone `s_nop`. **An empty asm with `~{memory}` is not a fence** —
it constrains the compiler only: `inline-asm/classes.md ## Class 2 — scheduling control`.

## Class 4 — synchronization

Reader-drain marker, `s_waitcnt vmcnt(N)`, `buffer_wbl2` / `buffer_inv`, dependency-carried drain:
`inline-asm/classes-sync.md ## Class 4 — synchronization`.

## Class 4b — protocol blocks

Bodies with branches, labels or an EXEC role-split (multi-GPU acquire/release):
`inline-asm/classes-sync.md ## Class 4b — protocol blocks`.

## Class 5 — cross-lane and wave-collective

Ballot / elect / rank / DPP / `permlane`; the preconditions attach to the construction:
`inline-asm/classes-lane-register.md ## Class 5 — cross-lane and wave-collective` (Class 6,
register-class control, is in the same file).

## What it costs you, on every class

The optimizer stops seeing through the block; portability is per-mnemonic and fails late; durability
across Triton upgrades is per cell: `inline-asm/costs.md`.

### Deciding the bins

Bin boundaries go where a derived parameter or a precondition changes: `inline-asm/shape-keying.md ## Deciding the bins — a procedure`.
