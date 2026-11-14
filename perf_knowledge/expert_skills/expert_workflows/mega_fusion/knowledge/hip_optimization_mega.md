# HIP: what mode=mega adds to `hip_optimization.md`

Read `hip_optimization.md` (shared kernel_workflow knowledge) first.

## When the hoist costs more than the recompute (LICM → spill)

Loop-invariant code motion is normally free: compute once outside the loop instead of every
iteration. It stops being free when the hoisted values have to **stay live across the whole loop**.
Each one occupies a register for the loop's entire duration, and past the allocator's budget they go
to scratch — so the kernel now spends a `scratch_load` per iteration to avoid an arithmetic op it
could have redone in a few VALU cycles. The compiler made a locally correct trade with no visibility
into the register pressure the rest of the loop would create.

The shape that triggers it: a **peeled or drained final iteration** whose addresses constant-fold to
compile-time values. Folded constants look maximally hoistable, so a peeled tail with N folded
addresses can inject N new function-entry invariants into a loop that was already at its register
ceiling. Measured on this workflow: 32 constant-folded LDS addresses in a peeled final-K drain,
hoisted to function entry, then spilled.

**Detect it in the ISA metadata, not in the source.** Compare across the change:
- `scratch` / private-segment size and the spill counts in the kernel descriptor — a jump here with
  no new local arrays is the signature,
- `scratch_load` / `scratch_store` appearing *inside* the hot loop,
- `.vgpr_count` at the ceiling while occupancy drops a step.

**The fix is to make the value LICM-ineligible without changing it.** Pass it through an identity
inline-asm — `v_mov_b32 $0, $1` — that also takes a per-iteration runtime value (e.g. a `buffer_load`
result) as an operand it never textually uses. The compiler cannot prove the asm is loop-invariant,
so it will not hoist it; the emitted instruction is a register move, and the value is bit-identical.

Two obligations if you use this, because it is a load-bearing dependency on one compiler version's
behaviour and nothing in the source says so:

- **Verify the effect in the emitted ISA, before and after** — spill counts and loop body — and quote
  both in the report. The source diff is not evidence; the whole mechanism lives below it.
- **Re-check after any toolchain bump.** A later LICM that sees through the anchor silently restores
  the spill; an earlier scheduler that never hoisted makes the anchor pure overhead. Record the
  compiler version alongside the ISA deltas so the next run can tell which happened.

This is the same failure family as `distributed_fusion.md` Lever 6b (a wait the compiler is allowed
to move is not a wait): in both cases the source expresses an intent the compiler is under no
obligation to honour, and the only place the intent is either kept or broken is the ISA.
