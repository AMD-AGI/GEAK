# Field guide: reverse index, traps, idiom catalogue

Part of `../inline-asm-reference.md`. Use this when you are *reading* asm someone else wrote, or
checking your own before you commit it.

## Reverse index: I see X in an unknown kernel

**Left:** what is on the page in front of you. **Then:** what the author was doing, what must already
be true for it to be correct, and the one thing to go and check.

### By constraint string

| You see | Intent | Invariant being held | Check this |
| --- | --- | --- | --- |
| `asm=""`, `"=v,0"` | pin the value into a VGPR at this point; zero instructions | the **returned** value is what is consumed downstream — the original binding is dead | is the result rebound (`acc = gl.inline_asm_elementwise(...)`)? If discarded, the site is dead |
| `asm=""`, `"=s,0"` | assert wave-uniform, keep it in an SGPR | the operand contains no lane index — **unchecked** | trace the operand: `program_id`, constexpr, `static_range` index → fine. `arange` → bug |
| `asm=""`, `"=v,=v,…,0,1,…"` with N outputs | pin N adjacent registers as one unit | the N slots cover the **whole** per-thread fragment | N is `Σ S`, **not** `pack` — on a sub-4-byte `dtype` they differ, and too few slots fences a prefix silently (`shape-keying.md ## Deriving the parameters`) |
| 64-bit `dtype` with a tie | force an **aligned register pair** | a 64-bit-wide operand — pointer, or two bit-packed f32 — not integer arithmetic | a 64-bit element is **one** slot; look for `<< 32` / `>> 32` or `_pack_*_pairs` bracketing it |
| `~{memory}` on an empty asm | block **compiler** motion of memory ops across the point | **NOT a fence.** Real ordering is `gl.barrier()` / `wait_group` elsewhere | its absence on an otherwise identical sibling is a real semantic difference |
| `is_pure=True` **and** no `~{memory}` | the site makes no memory-ordering claim at all | — | if you are relying on it for ordering you are relying on nothing |
| `is_pure=True` **and** `~{memory}` | contradictory — purity permits exactly the motion the clobber forbids | — | One of the two is wrong. A lint rule, not a judgement call |
| `is_pure=False`, no clobber, no side effect | prevent CSE/hoisting of the identity itself | the site's *position* matters, its value does not | two adjacent sites differing only in `is_pure` differ deliberately |
| `"=v,0"` on a **non-empty** body | the body reads and writes one register (reduction chain, `writelane`) | — | rewriting to `"=v,v"` needs `$0`→`$1` in the first body line |
| `"=v,=v,0,1,v,v"` with only **two** args | identity on two values *plus two artificial extra reads* — a false dependency edge | the site sits between the extra operand's producer and its real consumer | constraint count ≠ arg count is correct here, not broken |
| a `v` input with **no** output slot | a pure **dependency anchor** — only its readiness is used | the anchored value's producer is the thing being ordered against | its type must be integer/bitcast, never float into a counter |
| `"={scc}"` as an **output** | the result is delivered in the condition code, not as a value | nothing clobbers SCC between the asm and its consumer | the `dtype` element will be `gl.int1` |
| `~{scc}` / `~{vcc}` | the body names those architectural registers | — | body says `vcc` and the list does not → the code is wrong |
| `~{s0}`, `~{s1}`, … | the body writes **physical** SGPRs by name | the allocator must be told | omission silently corrupts whatever was there |
| `=&{v0}`…`=&{v11}` | **absolute physical assignment** so the body can write `v[0:3]` | nothing else may use those registers | non-portable; survives no change in `pack` or operand count |
| `=&` early clobber | the output is written before all inputs are read | — | mandatory for any body with ≥2 lines reusing `$0` as a temp |
| `n` constraint letter | a compile-time immediate folded into the encoding | the arg is a `gl.constexpr` | it fails to compile if not — a good property |
| constraints from a `constexpr_function(pack)` | a width-parameterized tie | the generator's argument must be the **slot count** `S`, which equals `pack` only for 4-byte-or-wider types | read the generator: does it append `~{memory}`? Some do, some do not. And check `S` against the `dtype` |
| constraints chosen by `if/else` on a constexpr | **shape-keyed** register grouping | `pack` is switched by the **same** constexpr | they must agree; check both lines |
| a **clobber list** chosen by `if/else` on a tile constant | shape-keyed **physical register reservation** | the asm text and an operand move with it | three coupled edits, not one |
| `dtype` as a **tuple** | multiple distinct outputs | the **tuple**, not `pack`, sets the output count — inputs start after `Σ S` over the tuple | the commonest transcription error on this page |
| `dtype` **omitted** | the output is a throwaway | — | seen on bare `s_setprio`; adding one changes the output register class |

### By mnemonic

| You see | Intent | Invariant being held | Check this |
| --- | --- | --- | --- |
| `s_setreg_imm32_b32 hwreg(HW_REG_MODE, 23, 1), 0`/`, 1` | narrow-float conversion mode save/restore | **wave-persistent and sticky**; a matched pair, the `0` dominating the `1` on every path | find the restore — it is legitimate for there to be none, if a launch partition guarantees it |
| `s_setreg` with a `v_mov_b32 $0, …` tail | the mov is not arithmetic — it fills a mandatory output | the token is routed into what must not be hoisted | — |
| `s_waitcnt lgkmcnt(0)` next to an `async_copy.wait_group` | drain **LDS readers** — a different obligation from copy completion | the slot is wave-private, or a `gl.barrier()` is also present | both, unless the source explains the elision |
| `s_waitcnt vmcnt(0)` | drain-all before a dependent read or a release | the read is genuinely dependent | usually paired with `~{memory}` |
| `s_waitcnt vmcnt(N)`, N ≠ 0 | hand-rolled multibuffering; the author is counting copies | **exactly** N VMEM ops in flight here | look for a `static_assert` pinning legal counts *and* a class-2 marker stopping copy motion |
| `s_waitcnt vmcnt($1)` with `n` | the same, with the depth as a folded immediate | the count equals the exact per-panel VMEM instruction count | the source should derive it; a `vmcnt(<digits>)` text census is blind to this form |
| `s_waitcnt … expcnt(0)` | also drain exports | — | only seen inside full multi-GPU barriers |
| `s_waitcnt` with **tied inputs** | the **completion half** of a split issue/complete pair | the issue site is a separate block with no wait; the tie is the only link | find the partner before judging either |
| `buffer_wbl2 sc0 sc1` | cross-device **release** | a peer is about to read this | there must be a matching acquire |
| `buffer_inv sc0 sc1` | cross-device **acquire** after a successful poll | payload reads follow in program order | a *terminal* wait with no reads after it correctly omits it |
| `global_load_dword … sc0 sc1` inside a backward branch | a **spin poll** on a peer counter | `sc0 sc1` is mandatory — a cached poll never terminates | — |
| `global_atomic_add … sc1` | peer arrival notification | preceded by a release; usually EXEC-masked | — |
| `s_sub_u32`/`v_subrev_u32` then a **signed** compare in a poll | wrap-tolerant epoch comparison | epoch steps small relative to 2³¹ | an unsigned compare deadlocks on rollover |
| `s_cbranch*` + `1:`/`2:` or `${:uid}` labels | a **protocol block** — the binding cannot express it at all | lane predication passed in as an operand under an explicit layout | `~{scc}`, `~{vcc}`, `~{memory}` all declared; `=&` on outputs |
| `${:uid}` in a label | per-instantiation unique label | — | numeric `1:`/`1b` collide if two blocks inline into one function |
| `s_and_saveexec_b64 $0, …` … `s_or_saveexec_b64 $0, $0` | run the body on a lane subset | `$0` is an `=&s` **EXEC save slot**, not data | dropping the restore corrupts EXEC permanently |
| `s_cbranch_execz` before a side effect | skip entirely when no lane qualifies | the body touches peer memory non-participants must not touch | — |
| `.if $8 == 0 … .endif` | **assembler-level** conditional on an `n` operand — no runtime branch | the operand is `n`-constrained | — |
| `s_setprio 1` … `s_setprio 0` | issue-priority bracket around a phase boundary | balanced, and not escaping through a branch | at least five different pinning mechanisms exist; and a bracket may be asymmetric on purpose |
| `s_nop N` before a DPP / `permlane` read | ISA **hazard cover**, not pacing | gfx950 needs two cycles VALU→DPP | never delete — unless the block interleaves independent streams, which substitutes for it |
| `s_load_dword(x2/x4/x8)` in asm | force a **scalar** load of wave-uniform data | the address has no lane component — **unchecked** | trace the address; a matching `lgkmcnt(0)` exists, possibly at another site |
| `s_or_b64`/`s_cselect_b*`/`s_bitcmp0_b64` | a predicate computed in the scalar unit | `~{scc}` declared, or SCC survives to the consumer | the "row active" idiom encodes the predicate as a **sign bit** |
| `v_readfirstlane_b32` | "this value is wave-uniform — give me an SGPR" | it really is uniform; **nothing checks it** | find the proof: a `gl.sum`, a gather, a broadcast constant |
| `v_readlane_b32 $0,$1,$2` | extract one chosen lane | the index is in `[0,63]` **and** uniform (`s`-constrained) | look for `gl.minimum(x, 63)` or an in-body `s_and_b32 $1,$1,63` |
| `v_readlane_b32 $0,$1,63` | terminates a full-wave DPP ladder | 64-lane wave; the ladder left the total in lane 63 | — |
| `v_writelane_b32` with `n` | insert a scalar into a compile-time-constant lane | the slot is a `gl.constexpr`, typically a `static_range` index | the destination is self-tied; the tie digit's *position* renumbers `$n` |
| `v_cmp_*_e64` with `=s` and `dtype=gl.uint64` | **ballot** — a 64-bit lane mask as a first-class value | `e64` (not `e32`) so it does not write `vcc` | look for a lane-0 gather right after |
| `v_cmp_*_e32 vcc, …` | ballot via `vcc` by name | `~{vcc}` declared | the `=&s` form is the safer spelling |
| `s_ff1_i32_b64` after a ballot | **lowest set lane wins** — deterministic tie-break | empty mask returns −1 and is guarded | a correctness device, not an optimization |
| `s_bcnt1_i32_b64` | population count of a mask | the result is a per-lane broadcast | collapse with `gl.sum(gl.gather(…))` |
| `v_mbcnt_lo` + `v_mbcnt_hi` | this lane's **rank** among set lanes — compaction / prefix | both must be present: `lo` seeds, `hi` accumulates | — |
| DPP `row_ror:`/`row_shr:`/`quad_perm:`/`row_half_mirror`/`row_mirror` ladder | butterfly reduction; **step count = log2(width)** | reduction axis mapped contiguously onto lanes; `=&v`; `s_nop 1` between steps | the `BlockedLayout`/reshape above the call is part of the contract |
| `row_bcast:15 row_mask:0xa` + `row_bcast:31 row_mask:0xc` | the canonical **full 64-lane** ladder | those two masks are required | `0xf` there is a wrong answer |
| `bound_ctrl:1` on a DPP max | out-of-range lanes read 0 | the identity **is** 0 — unsigned data, or a positive floor seed | sibling kernels differ on this and the difference is `not established` |
| `v_permlane16_swap_b32` | exchange a **register-index bit with a lane-index bit** | a matching `reshape`/`permute` is the software half | `s_nop 1` **before and after** is mandatory |
| `ds_swizzle_b32 … offset:0x40NN` | fixed lane-permutation exchange | the `s_waitcnt lgkmcnt(0)` is **inside the same body** | the swizzle constant must match the `^ k` math on the fallback branch |
| `v_rcp_f32` | hardware approximate reciprocal | a Newton step and an infinity guard follow **in Python**; sometimes a 2⁻⁶⁴ pre/post-scale | refinement comes *after* the rescale |
| `v_exp_f32` / `v_log_f32` | hardware transcendental — **base 2** | the caller already multiplied by log2(e) | passing a natural-log argument is silently wrong |
| `v_div_fixup_f32 $0,$1,$2,$3` | repair IEEE edge cases of an rcp-based divide | `$1`=quotient, `$2`=denominator, `$3`=numerator | not commutative |
| `v_med3_f32 $0,$1,-448,448` | a symmetric clamp in one instruction | NaN already removed upstream | the `±448` must match the FP8 format |
| `v_max3_u32`/`v_med3_u32`/`v_min3_u32` | branch-free sorting network | keys are **unsigned-comparable** | look for an encode step |
| `v_max_f64`/`v_min_f64` on `uint64` | borrowing the FP64 comparator for 64-bit unsigned ordering | all operands are normal-double bit patterns | nothing checks it; find the encoder |
| `v_max_u32` on float bits | max-of-bits == max-of-magnitude | only for non-negative floats | find where the sign was stripped |
| `v_dot2_f32_bf16` | 2-wide packed-bf16 dot-accumulate on the VALU | two bf16 of the reduction axis share one 32-bit lane slot | operands loaded as `uint32`; check the `size_per_thread` |
| `v_pk_*_u16` with `op_sel:`/`op_sel_hi:` | packed 16-bit SIMD-within-register, possibly **horizontal** | both halves are meaningful | `op_sel` decides which half feeds which operand — reads as a no-op self-op |
| `v_pk_fma_f32` with `dtype=gl.uint64` | packed 2-wide FP32 FMA | the `uint64` is a register-pairing device | the manual pack/unpack pair is load-bearing |
| `v_cvt_scalef32_pk_fp8_f32` | two f32 → a packed FP8 pair in one instruction | a layout with adjacent elements per thread, a split, and a `uint16` pointer reinterpretation | heavy caller-side setup |
| `v_mad_u32_u24` | 24-bit address math in one instruction | both multiplicands < 2²⁴ | usually unasserted; the result often carries an alignment annotation that inherits the assumption |
| `v_lshl_or_b32` / `s_lshl3_add_u32` | fused shift-and-combine address math | the `or` equals `+` only if the bit ranges do not overlap | usually unstated |
| `v_ffbl_b32` | lowest-set-bit index (slot allocation) | all-zero input returns −1 | callers cast to `uint32` and treat it as "large" |
| `\|$2\|` source modifiers | free absolute-value operand modifier | — | not exposed by the binding |

## Traps, ranked by how likely you are to hit one

Ranked by *how likely a competent engineer is to walk into it*: frequency of the idiom × silence of
the failure × how natural the wrong move is. **The first six are near-certain on any real editing
pass.**

1. **Reading `~{memory}` on an empty asm as a fence, then deleting the real barriers.** Empty tied
   blocks are the commonest shape there is, and the clobber constrains the *compiler*, not the
   hardware. The dangerous direction is removing the `gl.barrier()` pair next to one.
2. **Deriving the slot count from `pack` on a narrow `dtype`.** `pack` equals the number of `=v`
   slots only when the element is four bytes or wider; below that the count is
   `ceil(pack × sizeof / 4)`. Hand a `constexpr_function` generator `pack` instead of the slot count
   and it ties half the fragment, **compiles clean, and changes no answer** — the liveness or
   ordering property you bought silently covers a prefix. Where a narrow `dtype` is in play, write
   the strings out by hand under a `static_assert` rather than generating them.
3. **`pack` and constraint arity drifting apart for the ordinary reason.** One number derived twice —
   once at `pack=` and once inside the generator. Derive it once and pass the same expression to
   both. A hand-typed wide tie with no assertion is the version that rots first, and an expression
   like `// 256` hides a 4-wave × 64-lane launch assumption written nowhere else.
4. **Operand numbering.** `$n` counts *slots*, outputs first, so a 3-tuple `dtype` makes the first
   input `$3` and a `pack=8` bf16 output makes it `$4`; a tie digit's *position* in the list is not
   its meaning; and an argument may legitimately appear twice in two register classes.
5. **`is_pure` flipping meaning between visually identical sites.** `"=v,0"` with `is_pure=False` and
   with `is_pure=True` are the same string with different compiler contracts: `True` permits CSE,
   motion, and deletion-when-unused. Normalizing `is_pure` across a file converts hard fences into
   hints, or adds false barriers. The one spelling that is mechanically wrong is `is_pure=True`
   **with** `~{memory}`.
6. **Deleting "unused" arguments.** A `s_waitcnt` carrying a long list of payload anchors, or an
   input with no output slot, **is** the ordering statement. Removing the arguments compiles clean
   and destroys it.
7. **Base-2 transcendentals.** `v_exp_f32` / `v_log_f32` require the caller to have pre-multiplied by
   log2(e). Nothing in the asm hints at it, and the result looks plausible.
8. **Auditing half a split-phase pair.** An issue block with no wait guarantees nothing; the two
   halves may be linked only by a tie. Reordering or dropping the second call reads a register whose
   load has not landed.
9. **Stripping an `s_nop` that looks like padding.** It is a VALU→DPP hazard cover, and a `permlane`
   swap needs one *before and after*. The mirror error: removing one of the interleaved independent
   streams from a block that has **no** `s_nop`s precisely because the streams supply the spacing.
10. **Not rebinding the result.** `acc = gl.inline_asm_elementwise(...)` — drop the assignment and
    the whole site is dead. Every M0 site depends on this.
11. **Treating the `else:` prefilter as the delete test.** "Sits in an `if`-body whose `If` has an
    `else`" is AST-decidable, and its precision is not 1: it will hand you machine-state writes whose
    deletion is a wrong answer, and sites where both arms are asm. Run the three confirmation checks
    (`costs.md ## Durability and rollback`) before believing any candidate.
12. **Transplanting a shape-keyed site.** Asm text, opcode, constraint string, `pack`, clobber list
    and comparison polarity can all be keyed on tile constants, and a single site may key three at
    once. See `shape-keying.md`.
13. **A hand-counted `vmcnt(N)` decoupled from the marker that protects it.** Removing the
    copy-issue marker lets the compiler move copy issues and turns a correct `vmcnt(4)` into an
    incorrect one. **The coupling is the hazard, not either site.**
14. **`s_ff1_i32_b64` returning −1, guarded two incompatible ways** — inside the body versus outside.
    Mixing the conventions double-guards or under-guards.
15. **Uniformity asserted and never checked.** `"=s,…"` inputs, `v_readfirstlane_b32` and every
    scalar-load idiom take lane 0 silently if the value is divergent. Find the proof; there is no
    runtime check.
16. **The machine-mode bit is sticky, and a legitimate site may never restore it** — relying entirely
    on a launch partition. Lifting such a helper elsewhere gives silently wrong numerics.
17. **EXEC save slots mistaken for data.** In `s_and_saveexec_b64 $0, vcc` … `s_or_saveexec_b64 $0,
    $0`, `$0` is an `=&s` slot. Dropping it, tying it, or skipping the restore corrupts the wave's
    EXEC permanently.
18. **Numeric local labels colliding on inlining.** `1:` / `1b` work until two such blocks land in
    one function. `${:uid}` is the safe form.
19. **"Normalizing" DPP modifiers.** `row_mask:0xa` / `0xc` on the two `row_bcast` steps are
    required, not sloppy. And `bound_ctrl:1` under a `max` is safe only because of a positive floor
    seed.
20. **"Fixing" a `buffer_inv` asymmetry, or assuming `quad_perm` legality is a property of the asm.**
    A terminal wait with no payload reads after it correctly omits the invalidate — both directions
    of that "fix" are bugs. And `quad_perm` legality belongs to the `BlockedLayout` / `SliceLayout`
    **above** the call: change the layout and it compiles and decodes garbage. Finally, a
    dead-*looking* site is often not dead — a discarded output and dead asm text can sit on top of
    input constraints that are the entire content.

## Idiom catalogue

Twenty-two recurring shapes, one canonical name each. Aliases are given because these get named
differently in different write-ups — the alias column is the join key.

| Canonical name | Aliases seen | Shape |
| --- | --- | --- |
| **Tied identity** | instruction-free tied fence; tied-register identity; empty-asm tied identity | `("", "=v,0", [x])`, widened to `"=v,=v,…,0,1,…"`, optionally `~{memory}` |
| **Generated constraint string** | fence generator; `_identity_constraints`; `_fence_constraints`; `_dependency_constraints`; `_priority_constraints` | a Python/`constexpr_function` helper joins `["=v"]*n + range(n) [+ "~{memory}"]`; `n` is the **slot** count |
| **Register-file pin** | `"=s,0"` scalar bind; `_scalar_tile_base`; `_scalar_group_base` | empty asm, `=s` output, tied uniform input |
| **Register-pair device** | `_pair_accumulator`; 64-bit tie | pack two f32 into `uint64`, tie, unpack |
| **Physical-register asm** | named-VGPR asm; `~{s0}` clobber block | `"=&{v0}"…`, or `~{s0}`…`~{s7}` with `s_load_dwordx8 s[0:7]` |
| **Untied memory-clobber marker** | `_copy_issue_fence`; bare empty-asm issue fence; `_peer_read_fence` | `("", "=v,~{memory}", [])` — no tie, so only memory motion is constrained |
| **Drain marker** | naked waitcnt fence; wave-drain marker with a throwaway output | `"s_waitcnt lgkmcnt(0)\n v_mov_b32 $0, 0"`, `"=v,~{memory}"`, `args=[]` |
| **Hand-counted partial drain** | parameterized partial drain; `vmcnt(N)` multibuffer | `"s_waitcnt vmcnt($1)"`, `"=v,n,~{memory}"`, count derived per panel |
| **Dependency-carried LDS drain** | `_consume_lds_operand`; `_operand_ready` | a tie on the LDS read's result + `~{memory}`; the backend inserts the wait |
| **Dependency-anchor operand list** | payload anchors | a wait whose operands exist only to create edges — up to 19 of them |
| **Split-phase scalar load** | issue/wait metadata pair; `_issue_*`/`_complete_*`; `_issue_metadata`/`_wait_metadata` | issue block with no wait; a separate tied block supplies the wait |
| **Uniform scalar load** | `s_load` metadata loader; `_load_uniform_scale` | `"s_load_dword $0,$1,imm\n s_waitcnt lgkmcnt(0)"`, `"=s,s"`; mnemonic chosen from the pointer element type |
| **MODE-bit bracket** | FP8 mode save/restore; MODE-register poke; `_begin_native_fp8`/`_end_native_fp8`; `_ieee_pid`/`_fp8_scale` | `s_setreg hwreg(HW_REG_MODE,23,1), 0` … `, 1`, each anchored to a live value |
| **Priority bracket** | `s_setprio` phase marker; `_set_priority`; `_set_wave_priority` | `s_setprio 1` … `s_setprio 0`, pinned five different ways |
| **DPP butterfly ladder** | DPP butterfly reduction with `s_nop` spacing; `_q_group_max`; `_wave_maximum_pair` | `s_nop 1` + a DPP-modified `v_max`/`v_add` per step; step count = log2(width) |
| **permlane16 bit swap** | register/lane bit exchange; `_store_snapshot_eight` | `s_nop 1` + `v_permlane16_swap_b32` ×N + `s_nop 1`, all operands tied, with a matching Python permute |
| **Ballot → elect → readlane** | ballot → FFS; `_elect_expert`; `_wave_winner`; `_stable_expert_four` | `v_cmp_*_e64` → `s_ff1_i32_b64` → `v_readlane_b32`, bracketed by `convert_layout(…, assert_trivial=True)` and a `gl.sum(gl.gather(…))` collapse |
| **Lane extract / inject** | readfirstlane / readlane / writelane | `v_readfirstlane_b32` = assert uniform; `v_readlane_b32 …, idx` = take a named lane; `v_writelane_b32` = place a scalar into one lane |
| **Transcendental with a caller-side contract** | hardware transcendental + software correction; fast-math instruction selection | `v_rcp_f32` / `v_exp_f32` / `v_log_f32` / `v_div_fixup_f32`, `"=v,v"`, `is_pure=True`, with the contract in Python |
| **Packed-16 codec** | BF16 transport codec; `_pack_tile` | chains of `v_pk_{add,sub,min,max,mad}_u16` + `v_bfi_b32` / `v_alignbit_b32` / `v_perm_b32`, with `quad_perm` inside |
| **Protocol block** | multi-GPU lock protocol; cross-GPU release/poll/acquire; class 4b | predicate-and-skip, labelled spin on a `sc0 sc1` load, `buffer_inv`, EXEC save/restore |
| **EXEC bracket** | `s_and_saveexec` pair; `_notify_peers` | `s_and_saveexec_b64 $0, <mask>` … `s_or_saveexec_b64 $0, $0` with `$0` as an `=&s` save slot |
