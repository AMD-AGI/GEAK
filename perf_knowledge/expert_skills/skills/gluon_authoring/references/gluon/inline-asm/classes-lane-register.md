# Classes 5 and 6: cross-lane collectives, and register-class control

Part of `../inline-asm-reference.md`. Classes 1–3: `classes.md`; 4/4b: `classes-sync.md`.

## Class 5 — cross-lane and wave-collective

Ballot, find-first-set, `v_readlane`/`v_writelane`, `v_mbcnt` rank, `v_bcnt` popcount, DPP ladders,
`permlane` swaps — no tensor-level spelling. The asm assumes **element order is lane order**; earn that
per construction:

- **A — 64-element ballot tensor** (ballot/elect/rank): length == wave size, `size_per_thread=1`,
  `warps_per_cta=1` **and** `num_warps=1`, `pack=1`, element order == lane order. Assert it — violating
  any compiles cleanly and is wrong at an unrun shape:
  ```python
  gl.static_assert(x.shape[0] == 64)
  local = gl.convert_layout(local, gl.BlockedLayout([1], [64], [1], [0]), assert_trivial=True)  # the guard
  ```
- **B — wider predicate tensor used only as an operand**: `gl.arange(0, 256, layout=gl.BlockedLayout([1], [64], [4], [0]))`
  compared in-body (`s_cmp_lt_u32 $0, 64`) — no bijection needed (protocol-block predication).
- **C — packed swap, `pack > 1` required**: `v_permlane16_swap_b32` with matching tied operands plus the
  software half — a matching `reshape → permute`, a `gl.DistributedLinearLayout`, and
  `convert_layout(..., assert_trivial=True)`; the permute must match the hardware swap exactly.

### Invariants the asm cannot check for you

- **Hazard spacing:** `s_nop 1` between DPP steps (the gfx950 wait-state table lists VALU-writes-VGPR →
  DPP-reads-it as 2 wait states) unless the block interleaves independent streams that supply it; the
  production blocks seen also put `s_nop 1` before *and* after `permlane` (no `permlane` row in the in-pack
  table — check `amd_isa.py --arch gfx950 hazard` / `waits`).
- **Full 64-lane ladder** = `row_shr:8/4/2/1` + `row_bcast:15 row_mask:0xa` + `row_bcast:31 row_mask:0xc`,
  then `v_readlane_b32 $0, $1, 63`. The masks are **not** `0xf`; step count = log2(group width).
- **`bound_ctrl:1` makes out-of-range lanes read 0** — fine for unsigned max or a positive-floor-seeded
  max, wrong for signed data. (Worked example, if aiter is available:
  `aiter/ops/triton/_gluon_kernels/gfx950/gated_delta_net/fused_gdn_decode_qkvz.py`.)
- `"=v,0"` on a **non-empty** body reads and writes one register; switching to `"=v,v"` needs `$1` in the first line.
- **`s_ff1_i32_b64` returns −1 on an empty mask**; guarded inside (`s_and_b32 $1, $1, 63`) or outside
  (`gl.minimum(first_lane, 63)` + re-check) — mixing conventions double- or under-guards.
- A ballot/popcount result is a per-lane broadcast: collapse with `gl.sum(gl.gather(x, zero, 0), 0)`.
- Electing the lowest set lane is a **determinism device** (stable top-k tie-break), not an optimization.

**Verify:** mnemonics in the `.s` + a test whose answer depends on the lane mapping (a ballot over a mask with known population).

## Class 6 — register-class control

The constraint string is the instruction (asm usually empty). **Most empty tied sites are Class 2
dependency shaping, not this**; a site is Class 6 (intent I-C) only when deleting it changes which
instructions are selected downstream (`classifying.md ## Axis I — intent`).

- **Register file:** `"=s,0"` asserts wave-uniform → SGPR, so a following VMEM uses scalar base +
  vector offset (`"=v,0"` = VGPR form; one-instruction variant `s_mov_b64 $0, $1` with `"=s,s"`). The
  operand must trace to `program_id` / constexpr / `static_range` index — from `gl.arange` it is a lie: on
  gfx950 / 3.8.0 LLVM prints `illegal VGPR to SGPR copy` yet a binary is produced, and the result was garbage
  (not lane 0's value) when run. **For MFMA accumulators the builtin's register-class argument, where
  the build has one, is the supported route** (the AMD backend now steers accumulators with an internal empty
  tied asm; no upstream test of the elementwise-door form was found). Default placement facts:
  `../../tile-programming/llvm-fn-attrs.md`.
- **Adjacency:** a 64-bit tied operand forces an **aligned register pair** — pack two f32 into `uint64`,
  tie, unpack (a pairing device, not integer math; check with `convert_layout(result, acc.type.layout, assert_trivial=True)`).
- **Physical:** `"=&{v0}"`…`"=&{v11}"` so the body can name `v[0:3]`, or `~{s0}`…`~{s7}` around `s_load_dwordx8 s[0:7]`.
  Least portable thing here: survives no change in `pack`, operand count or register budget.
- **A use-site can pin values it does not return:** `v_mov_b32 $0, 0` with `"=v,s,s,s,s,s,s,s,s,~{memory}"`
  forces eight inputs into SGPRs at that point while the function returns the pre-asm values. Comment it.

### `pack` and the constraint arity are one decision — and the relation is not identity

Slot count `S = ceil(pack × sizeof / 4)` for <4-byte types (input side: `inline_asm_elementwise` docstring;
the same holds for outputs) — full law: `shape-keying.md ### The arity law — pack, dtype and the constraint string are one decision`.
**A slot count that disagrees with Σ S(`dtype`) at the given `pack`** — too few or too many — aborts the
compile on Triton 3.8.0 / gfx950 with an LLVM assertion ("number of output constraints does not match"),
not a clean error. The silent case is a consistent (`pack`, slots) pair whose `pack` was derived wrong: a
`pack` above the per-lane element count is padded with `undef` operands, one below it splits the fragment
into several separate asm instances (compile-only check). Change `pack`, slot count and tie digits
**together**; a generator that takes `S`, not `pack`, keeps them in step:

```python
@triton.constexpr_function
def _identity_constraints(slots):          # NOTE: slots, not pack
    return ",".join(["=v"] * slots + [str(i) for i in range(slots)] + ["~{memory}"])

acc = gl.inline_asm_elementwise("", _identity_constraints(S), [acc], dtype=gl.float32, is_pure=False, pack=PACK)
```

Generators that look identical may differ in appending `~{memory}` (sometimes conditionally on a tile
flag) — read before reuse. `BM * BN // 256` hides a 4-wave × 64-lane launch assumption. Bitcast f32→i32
around the tie is deliberate — copy it with the tie. `pack` and the tie string are often switched by the
same shape flag (`"=v,=v,0,1"`/`pack=2` vs four slots/`pack=4`); re-derive on re-tile (`shape-keying.md`).

**Verify:** read register operands in the disassembly — a pin that did not take looks identical until something spills.
