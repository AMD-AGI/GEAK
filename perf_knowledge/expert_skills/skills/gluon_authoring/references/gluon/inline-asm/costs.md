# What it costs, how to verify it, and how to get rid of it

Part of `../inline-asm-reference.md`.

## What it costs you, on every class

- **The optimizer stops seeing through the block** (inputs materialized, scheduler works around it);
  `is_pure=False` additionally blocks CSE/hoisting of the block itself.
- **Portability is per-mnemonic and fails late**; record target / compiler / mode preconditions next to the kernel.
- **Durability is per cell** (table below); sites are often half of a pair (`PAIRED-WITH`) and per-shape
  artifacts (`shape-keying.md`).

## Verifying that it worked — write, compile, dump, check

0. **Trace** — catches your `gl.static_assert`s (no assertions ⇒ no information). It does not catch a runtime
   value on an `n` operand: on 3.8.0 / gfx950 that compiles and LLVM silently drops the block (step 2 catches it).
1. **Dump** — `kernel_workflow/scripts/kernel_tools/dump_ir.sh --variant <name> --out <ir_dir>` (pack shim
   `kernel_workflow/scripts/kernel_tools/dump_ir.sh`) → `<variant>.{ttgir,llir,amdgcn,s}`. **`TRITON_ALWAYS_COMPILE=1`** or you read a
   stale cache hit. Pin one Triton build (`../../tile-programming/compiler-contract.md`).
1.5. **Hash the normalized disassembly with vs without the site (no GPU)**; equal ⇒ inert, delete it. Also
   drop `.debug_*` sections — they embed the source filename, so a different-length filename fakes a diff.
2. **Count the mnemonic** against (sites × region instances) computed beforehand. Zero ⇒ (a) `is_pure=True`
   with an unread result, (b) result not rebound (`acc = gl.inline_asm_elementwise(...)` — every M0 site
   depends on it), (c) the constexpr branch folded away. Fewer ⇒ CSE'd. More ⇒ cloned/hoisted (harmful for an I-A bracket).
3. **Placement:** name the two instructions it must sit between before dumping. Floated ⇒ missing anchor:
   crossed memory op → `~{memory}`; crossed dependent arithmetic → no tie / tied result not consumed;
   `s_setreg` sank → anchor to a token (`classes.md ## Class 3 — machine state`); `s_setprio` moved → weak
   pinning (`classes.md ## Class 2 — scheduling control`); wait moved → someone removed an "unused" operand.
4. **M0 (nothing to count): verify by effect, as a diff vs the arm without it** — `"=s,0"`: consumer uses
   `s[...]` base; dependency drain: `s_waitcnt lgkmcnt` lands before the block and the next copy after
   it; liveness: `next_free_vgpr`, metadata counts, hot-loop `scratch_*` via
   `kernel_workflow/scripts/kernel_tools/asm_loop_audit.py`. A tie that moved spills the wrong way is to remove, not widen.
5. **Deletion probe:** leave the result unused, confirm the site vanishes (proves consumption is the anchor).
6. **Test the intent:** I-A — input that exercises the state (real overflow); I-B — determinism race-test,
   ≥~40 launches, `max|out_i - out_0| == 0` (`../../method/benchmark-hygiene.md ## Determinism race-test (async / barrier / pipeline / layout changes)`), at the largest shape in the bin;
   I-C — numeric diff vs `else:` on edge data (inf, empty mask, denormal); I-D — **bit-identical** or your classification is wrong.
7. **Record** target, compiler, launch geometry, shape bin, expected mnemonic count — the count is your CI regression test.

## Worked derivation — a shared-memory reader drain at a new shape

bf16 GEMM, `num_warps=4`, one wave-private LDS slot per operand, 16 bf16 per lane per read; the `.s` shows
the next `buffer_load_to_shared` issuing before the `ds_read_b128`s retire (no `lgkmcnt` between). Order
without data flow, consumer-side ⇒ **I-B** (`wait_group` covers only the producer). Choose M0-tie (no count
to go stale; needs wave-private slot — assert it). `pack=16`; bitcast to `gl.uint16`; `S = ceil(16×2/4) = 8`
(**8 slots, not 16**); `is_pure=False`; no `=&`, no `~{scc}`.

```python
if DEPENDENCY_FENCE:
    words = gl.inline_asm_elementwise("", "=v,=v,=v,=v,=v,=v,=v,=v,0,1,2,3,4,5,6,7,~{memory}",
                                      [value.to(gl.uint16, bitcast=True)], dtype=gl.uint16,
                                      is_pure=False, pack=16)
    value = words.to(value.dtype, bitcast=True)      # the rebinding IS the anchor
else:                                                # fallback, also asm -- not a removal arm
    gl.inline_asm_elementwise("s_waitcnt lgkmcnt(0)\n v_mov_b32 $0, 0", "=v,~{memory}", [],
                              dtype=gl.int32, is_pure=False, pack=1)
```

Verify via step 4 (`s_waitcnt lgkmcnt` after the last `ds_read_b128`, before the next copy), step 5, and a
race-test. Bins: >1 wave sharing the slot breaks wave-privacy; an 8-element fragment gives `S=4` —
`gl.static_assert(PACK == 8 or PACK == 16)`. In plain Gluon on 3.8.0 the binding that emits this drain is `gl.barrier()` at `num_warps > 1` (heavier: it
adds a CTA barrier); at `num_warps=1` it emits neither the wait nor `s_barrier`. A ring deep enough that no
slot is refilled while it is still read avoids the obligation altogether.

## Durability and rollback — what survives an upgrade

| Cell | Survives a Triton upgrade? | Survives a target change? | Fails loudly? | Re-test when |
| --- | --- | --- | --- | --- |
| I-A / M1 machine state | usually (the text is the instruction; re-check anchoring) | **not guaranteed** | no — wrong numerics | target change; any lift |
| I-A / M3 protocol block | usually (re-check anchoring) | **not guaranteed** | hang yes, race no | target, world size, record layout |
| I-C / M1 instruction | usually | per mnemonic | no | binding upgrade (may become removable) |
| I-B / M1 `vmcnt(N)` | **not guaranteed** (the count depends on the compiler's issue order) | n/a | **no** | every compiler bump and re-tile |
| I-B / M0 dependency drain | re-check (delegates to the wait-count pass) | n/a | no | every compiler bump (step 4) |
| I-D / M0 dependency shaping | **not guaranteed**, and a loss is silent | n/a | no | every re-tile |
| any `phys` | not guaranteed | not guaranteed | no | any edit |

**After a Triton bump:** (1) the site still emits (step 2; a site that stopped emitting or hashes inert is
to be **removed**), (2) re-derive hand-counted wait depths, (3) re-check dependency drains (step 4), (4)
`hasattr(rocdl, '<name>')` / a new `gl.*` call may make an I-C site removable, (5) re-check generated
constraint arity. The call signature itself is unchanged 3.6.0–3.8.0 (upstream source at those tags).

### The one mechanical removal test — prefilter, then confirm

**An `else:` branch computing the same thing in plain Gluon ⇒ the site is I-C and deletable for
correctness.** The AST prefilter (site in an `if` with non-empty `orelse`) is not the test; confirm all
three: the `else:` computes the same values in **plain Gluon** (asm/asm arms do not qualify); the intent
rules return I-C (no `s_setreg*`, coherence-qualified/atomic op, cache maintenance —
`classifying.md ## Axis I — intent`); `PAIRED-WITH` is empty. Safe for correctness ≠ safe for the
schedule. Write the `else:` arm when you write the site: it is the rollback, the A/B switch and the oracle.
