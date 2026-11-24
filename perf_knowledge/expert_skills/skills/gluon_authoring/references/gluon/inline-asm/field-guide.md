# Field guide: reverse index, traps, idiom catalogue

Part of `../inline-asm-reference.md`. Use when *reading* asm someone else wrote, or checking your own.

## Reverse index: I see X in an unknown kernel

### By constraint string

| You see | Intent | Check this |
| --- | --- | --- |
| `asm=""`, `"=v,0"` | pin value in a VGPR here; 0 instructions | result rebound (`acc = ...`)? If discarded, no consumer sees the pin (with `is_pure=True` the site is deleted) |
| `asm=""`, `"=s,0"` | assert wave-uniform → SGPR with no instruction (vs `v_readfirstlane`) | operand from `program_id` / constexpr / `static_range` ok; from `arange` ⇒ LLVM prints `illegal VGPR to SGPR copy`, still emits a binary, and the result is garbage (GPU-checked gfx950 / 3.8.0) |
| `"=v,=v,…,0,1,…"` (N outputs) | pin N registers as one fragment | N = Σ S, **not** `pack` (sub-4-byte `dtype`) — `shape-keying.md ## Deriving the parameters — compute them, never copy them` |
| 64-bit `dtype` + tie | aligned register pair (pointer or two packed f32) | one slot; look for `<< 32` / `_pack_*_pairs` around it |
| `~{memory}` on empty asm | compiler-only motion barrier, not a fence (aiter `gfx950/attention/sparse_mla_fp8.py`) | real ordering is `gl.barrier()` / `wait_group` nearby; absence on a sibling is a real difference |
| `is_pure=True`, no `~{memory}` | no ordering claim at all | relying on it for ordering = relying on nothing |
| `is_pure=True` + `~{memory}` | contradiction | one of the two is wrong |
| `is_pure=False`, no clobber, no effect | stop CSE/hoist of the identity; position matters | adjacent sites differing only in `is_pure` differ deliberately |
| `"=v,0"` on a non-empty body | body reads/writes one register | `"=v,v"` needs `$0`→`$1` in line 1 |
| `"=v,=v,0,1,v,v"` with 2 args | identity + artificial extra reads (false dependency edge) | constraint count ≠ arg count is correct here |
| `v` input with no output slot | dependency anchor (readiness only) | integer/bitcast type, never float into a counter |
| `"={scc}"` output | result in the condition code | nothing clobbers SCC before the consumer; `dtype` `gl.int1` |
| `~{scc}` / `~{vcc}` | body uses them | body says `vcc`, list does not ⇒ wrong |
| `~{s0}`… / `=&{v0}`… | physical registers | non-portable; survives no `pack`/operand change |
| `=&` | output written before all inputs read | mandatory for ≥2 lines reusing `$0` |
| `n` | compile-time immediate in the encoding | arg must be constexpr; on 3.8.0 a runtime value is not rejected — LLVM silently drops the block (count the mnemonic) |
| constraints from `constexpr_function(...)` | width-parameterized tie | argument must be slot count `S`; does it append `~{memory}`? |
| constraints / clobbers chosen by constexpr `if` | shape-keyed grouping / physical reservation | `pack` (and asm text, operand) switched by the same flag |
| tuple `dtype` | multiple outputs | inputs start after Σ S over the tuple |
| `dtype` omitted | throwaway output (bare `s_setprio`) | adding one changes the output register class |

### By mnemonic

| You see | Intent | Check this |
| --- | --- | --- |
| `s_setreg_imm32_b32 hwreg(HW_REG_MODE, 23, 1), 0`/`1` | narrow-float mode bracket (sticky, wave-wide) | find the restore — absent is legal only under a launch partition |
| `s_setreg` + `v_mov_b32 $0, …` | mov fills the mandatory output | token routed into what must not float |
| `s_waitcnt lgkmcnt(0)` near `wait_group` | drain LDS **readers** (≠ copy completion) | slot wave-private, or a `gl.barrier()` present |
| `s_waitcnt vmcnt(N)`, N≠0 / `vmcnt($1)` + `n` | hand-counted multibuffer drain | `static_assert` on legal counts + a marker pinning copy issue |
| `s_waitcnt` with tied inputs | completion half of a split pair | find the issue half |
| `buffer_wbl2 sc0 sc1` / `buffer_inv sc0 sc1` | cross-device release / acquire | matching other half (terminal wait may omit `buffer_inv`); absent both is correct on uncached peer memory |
| `s_waitcnt vmcnt(0)` + `s_barrier` + `s_mov_b32 $0, 0`, result added into a store address | publish fence | the first store after it on every path is the flag |
| `s_memrealtime` + `s_trap` inside a poll loop | wall-clock spin bound | reserved SGPRs clobbered; the expiry value is reported, not consumed as data |
| `global_load_dword … sc0 sc1` in a backward branch | spin poll | `sc0 sc1` mandatory |
| `s_sub_u32`/`v_subrev_u32` + signed compare | wrap-tolerant epoch compare | unsigned compare deadlocks on rollover |
| `s_cbranch*` + labels / `${:uid}` | protocol block | `~{scc}`,`~{vcc}`,`~{memory}`, `=&`; a numeric label used but not defined in the block binds to another block's |
| `s_and_saveexec_b64 $0` … `s_or_saveexec_b64 $0, $0` | lane-subset body | `$0` is an `=&s` EXEC save slot |
| `.if $8 == 0 … .endif` | assembler-level conditional on an `n` operand | — |
| `s_setprio 1 … 0` | priority bracket | pinned five ways; may be asymmetric on purpose |
| `s_nop N` before DPP / `permlane` | hazard cover (VALU→DPP) | never delete (unless interleaved streams replace it); see the DPP example below |
| `s_load_dword(x2/x4/x8)` | scalar load of uniform data | address uniform (unchecked); matching `lgkmcnt(0)` |
| `s_or_b64` / `s_cselect_b*` / `s_bitcmp0_b64` | scalar predicate | `~{scc}`; "row active" encodes as sign bit |
| `v_readfirstlane_b32` | assert uniform → SGPR | find the uniformity proof |
| `v_readlane_b32 $0,$1,$2` / `…,63` | take one lane / end of a DPP ladder | index in [0,63] and `s`-constrained |
| `v_writelane_b32` + `n` | insert scalar into a constexpr lane | tie position renumbers `$n` |
| `v_cmp_*_e64` → `=s`, `dtype=gl.uint64` | ballot as a value (`e64` avoids `vcc`) | lane-0 gather after |
| `s_ff1_i32_b64` / `s_bcnt1_i32_b64` / `v_mbcnt_lo`+`hi` | lowest set lane / popcount / lane rank | −1 on empty mask; collapse broadcast; both mbcnt halves |
| DPP `row_shr`/`row_ror`/`quad_perm`/`row_mirror` ladder | butterfly reduction, steps = log2(width) | layout above the call is the contract; `s_nop 1` per step |
| `row_bcast:15 row_mask:0xa` + `row_bcast:31 row_mask:0xc`; `bound_ctrl:1` | full 64-lane ladder; out-of-range lanes read 0 | masks not `0xf`; `bound_ctrl` only for unsigned / positive-seeded max — worked: aiter `aiter/ops/triton/_gluon_kernels/gfx950/gated_delta_net/fused_gdn_decode_qkvz.py` |
| `v_permlane16_swap_b32` | reg-index bit ↔ lane-index bit | matching Python permute; `s_nop 1` before and after |
| `ds_swizzle_b32 … offset:0x40NN` | fixed lane permutation | `lgkmcnt(0)` in the same body; constant matches fallback `^ k` |
| `v_rcp_f32` / `v_exp_f32` / `v_log_f32` / `v_div_fixup_f32` | fast math with a caller contract | Newton + inf **and zero** guard (`classes.md`); base 2; `$1`=q,`$2`=den,`$3`=num |
| `v_med3_f32 $0,$1,-448,448` | one-instruction clamp | NaN removed upstream; ±448 matches the FP8 format |
| `v_max3/med3/min3_u32`, `v_max_f64` on `uint64`, `v_max_u32` on float bits | sorting network / borrowed comparator | keys unsigned-comparable / positive-normal / sign stripped |
| `v_dot2_f32_bf16` | 2-wide bf16 dot on VALU | operands as `uint32`; `size_per_thread` |
| `v_pk_*_u16` + `op_sel` / `v_pk_fma_f32` with `uint64` | packed 16-bit SWAR / 2-wide f32 FMA | `op_sel` routing; pack/unpack pair is load-bearing |
| `v_cvt_scalef32_pk_fp8_f32` | 2×f32 → packed FP8 | adjacent-element layout, split, `uint16` reinterpret |
| `v_mad_u32_u24` / `v_lshl_or_b32` | 24-bit / shift-or address math | operands < 2²⁴ / bit ranges disjoint — usually unasserted |
| `v_ffbl_b32` | lowest set bit (slot allocation) | −1 on zero input |
| `\|$2\|` | free abs source modifier | not exposed by the binding |

## Traps seen in asm-carrying kernels

1. Reading `~{memory}` on empty asm as a fence, then deleting the real `gl.barrier()`s.
2. Slot count from `pack` on a narrow `dtype` — too many slots; on 3.8.0 / gfx950 an LLVM assertion aborts the compile instead of a clean error.
3. `pack` and the generator's argument derived twice and drifting; `// 256` hiding a 4-wave launch assumption.
4. `$n` numbering: slots, outputs first; tie position ≠ meaning; duplicated args are legitimate.
5. Normalizing `is_pure` across a file (turns fences into hints or adds false barriers).
6. Deleting "unused" arguments that are the ordering statement.
7. Base-2 `v_exp_f32` / `v_log_f32` without the log2(e) pre-multiply.
8. Auditing half a split-phase pair.
9. Stripping a hazard `s_nop`, or removing one of the interleaved streams that replaced it.
10. Not rebinding the result (`acc = ...`) — the site no longer reaches its consumers.
11. Treating the `else:` AST prefilter as the delete test (`costs.md ## Durability and rollback — what survives an upgrade`).
12. Transplanting a shape-keyed site across shapes.
13. A hand-counted `vmcnt(N)` decoupled from the marker that pins copy issue.
14. Mixing inside/outside `s_ff1` −1 guards.
15. Uniformity asserted (`"=s"`, `v_readfirstlane`, scalar loads) and never proven.
16. Lifting an unrestored MODE write.
17. EXEC save slot treated as data.
18. Numeric labels colliding on inlining (a block referencing a label it does not define).
19. "Normalizing" `row_mask:0xa`/`0xc` or `bound_ctrl:1`.
20. "Fixing" a `buffer_inv` asymmetry; assuming `quad_perm` legality lives in the asm (it lives in the layout above); judging a discarded-output site dead when its input constraints are the content.

## Idiom catalogue

Helper names seen in production kernels → canonical idiom (grep key):
`_identity_constraints` / `_fence_constraints` / `_dependency_constraints` / `_priority_constraints` → generated tie string ·
`_scalar_tile_base` / `_scalar_group_base` → `"=s,0"` register-file pin · `_pair_accumulator` → 64-bit pair tie ·
`_copy_issue_fence` / `_peer_read_fence` → untied `("", "=v,~{memory}", [])` marker ·
`_consume_lds_operand` / `_operand_ready` → dependency-carried LDS drain ·
`_issue_*`/`_complete_*`, `_issue_metadata`/`_wait_metadata` → split-phase scalar load · `_load_uniform_scale` → uniform `s_load` + `lgkmcnt(0)` ·
`_begin_native_fp8`/`_end_native_fp8`, `_ieee_pid`/`_fp8_scale` → MODE-bit bracket · `_set_priority` / `_set_wave_priority` → `s_setprio` bracket ·
`_q_group_max` / `_wave_maximum_pair` → DPP ladder · `_store_snapshot_eight` → `permlane16` swap ·
`_elect_expert` / `_wave_winner` / `_stable_expert_four` → ballot → `s_ff1` → `v_readlane` · `_pack_tile` → packed-16 codec (`v_pk_*_u16`, `v_perm_b32`) ·
`_notify_peers` → EXEC bracket in a protocol block.
