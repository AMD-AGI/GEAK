# Low-Precision-Weight MoE on gfx950 (a16w4 / a8w4 / a4w4): measured levers and scaled-MFMA facts

The page for an expert GEMM that carries MXFP4 / fp8 weights (e2m1 or e4m3 + E8M0 group-32
scales) on gfx950. It is the short, verified page; `moe.md` holds the long-form survey and
`../tile-programming/low-precision.md` holds the dense-GEMM scaled-MFMA path. Everything below was
verified on Triton 3.8 / gfx950 (MI350X) unless a line says otherwise. gfx942 has no scaled MFMA and
no fp4 matrix instruction (CDNA3 ISA; no `cdna3.mfma_scaled` on 3.8.0); none of the scaled-MFMA facts
apply there.

## Decide the matrix route from the activation dtype, not the weight dtype

| activations | weights | route | why |
| --- | --- | --- | --- |
| bf16 / fp16 (a16w4) | MXFP4 | `cdna4.scaled_upcast` W -> bf16, then `cdna4.mfma` `[16,16,32]` | on 3.8.0 `mfma_scaled` asserts `e2m1` / `e4m3` / `e5m2` operands; a bf16 A cannot enter it without quantizing X (next paragraph) |
| fp8 (a8w4) | MXFP4 | `mfma_scaled(a, a_s, "e4m3", w, w_s, "e2m1", acc)` | mixed formats are legal; one instruction, no dequant VALU |
| fp4 (a4w4) | MXFP4 | `mfma_scaled(..., "e2m1", ..., "e2m1", ...)` | as above; fp4 x fp4 issues in half the matrix cycles of fp8 |

**a16w4 trap: quantizing X to MXFP8 so that `mfma_scaled` applies is a numerics change, not a
schedule change.** It removes the per-element convert VALU (the prefill tile gets markedly faster; decode
tiles are memory-bound and tie) but the activation rounding raises the output error by about an order of
magnitude, which fails a tight elementwise gate such as atol = rtol = 4e-2 with a 0.1 % outlier budget.
Whether it is admissible is decided by the task's correctness oracle (`../tile-programming/low-precision.md`,
correctness-gate precondition). Verify with a single-tile A/B: same tile, both routes, compare error and
the gate's outlier fraction, not only the norm error.

## Levers measured on a16w4 (scaled_upcast) expert GEMMs

These are the levers that paid on the one a16w4 MoE measured (gfx950); the order is that kernel's
measured payoff, not a general ranking. Each item names the signal that makes it relevant.

1. **Make the E8M0 scale expansion register-only**: 3-D linear layout `[BLOCK_K/32, 32, BLOCK_N]` split
   from `gl.to_linear_layout(DotOperandLayout(1, mfma, 8), ...)`, broadcast + reshape, then
   `gl.convert_layout(..., OUT, assert_trivial=True)`. Signal: scale `ds_write`/`ds_read` in the K loop.
2. **Per-bucket config**: `BLOCK_M` is fixed by the routing metadata, so sweep `num_warps` x `block_n` x
   `tiles_per_warp` jointly; re-sweep after every structural change, read `.vgpr_spill_count` per cell.
3. **Decode W cache policy per CTA**: streaming (`.cg` / nt) suited experts owning a single row block
   (weight read once) in the kernel measured.
4. **Under-filled decode launches: split-K with a cheap reduce.** This applies to gate_up at T=1.
5. **Combine: slot-major store + gather-reduce measured ahead of the atomic combine at decode.** See
   the next section.

## Combine structure (down projection with scatter + router weights)

- **An atomic combine into the token-major output lost at decode in the graph-replayed a16w4 / a8w4 MoE
  ops measured (gfx950).** It needs a zero-init of the output, which is a dispatch costing about what
  the reduce dispatch costs; the atomic epilogue was also slower than a plain store, and the slot->token
  inverse was not passed to the down call (it had to be searched in-kernel). Re-test if the output is
  already zeroed for free or the inverse map is available. With store + gather-reduce kept, the reduce
  itself is where the measured remaining time sat (wide unrolled rows, streaming cache modifier, fp32
  accumulate).
- Replicated MFMA layouts double-count atomics: keep `instr_m * tiles_per_warp <= BLOCK_M` or convert to a
  replication-free layout first.
- Atomic order is nondeterministic, and bf16 atomics round once per add. Gate on the task's oracle.

## Scaled-MFMA facts (gfx950, Triton 3.8)

### mfma_scaled k_width: use 16 whenever a scale is driven or formats are mixed

| condition | correct operand `k_width` |
| --- | --- |
| real E8M0 scales via `get_mfma_scale_layout`, `[16,16,128]`, any format pair | **16 only** |
| real scales, `[32,32,64]` | e4m3 x e4m3: 4 / 8 / 16; e2m1 or mixed: **16 only** |
| mixed `e4m3` x `e2m1`, scales `None`, either shape | **16 only** |
| same format, scales `None` | 4 / 8 / 16 / 32 all correct (e2m1 `k_width=64` wrong at `[16,16,128]`) |

Wrong values compile, run and return O(1) relative error: no assert and no NaN. The likely mechanism
is that `get_mfma_scale_layout(dot_layout, shape)` reads only `operand_index`, `instr_shape[0]`,
`tiles_per_warp` and `warps_per_cta`, not `k_width`. The scale's K groups match the operand's K
ownership only at 16 bytes per lane, and a uniform (`None`) scale hides the mismatch. So a `k_width`
copied from an unscaled fp8 kernel (the `e5m2` tutorial's 32) becomes wrong once the scales are
driven. Verify on one GPU with `python3 scripts/mfma_scaled_check.py`. It prints error and the emitted
`v_mfma` per (format, instr, scales, k_width).

### Other facts

- **`mfma_scaled(a, None, ..., b, None, ...)` still emits `v_mfma_scale_f32_*_f8f6f4`.** The frontend
  materializes a unit E8M0 splat. It is not the unscaled op. Verify: `grep v_mfma` in `h.asm["amdgcn"]`.
- **`get_mfma_scale_layout(..., scale_factor=32)` admits only 32** (3.8.0 asserts it), which is OCP MX
  group size. fp32 scales over 128-wide groups do not map onto the E8M0 hardware scale as they are;
  they stay on the accumulator descale route, unless they are powers of two and replicated per
  32-group (exact in that case; not tried here).
- **`scaled_upcast` on CDNA4 is native** (`v_cvt_scalef32_pk_bf16_fp4`, 2 fp4 per instruction). In a
  compute-bound a16w4 loop the per-K-step converts (a fixed count per K step, independent of `BLOCK_M`)
  add a near-constant cost that did not overlap the MFMAs (gfx950 / Triton 3.8.0, register-resident
  operands): they dominate at `BLOCK_M <= 32`, are a large addend at 64, and the bf16 MFMA rate takes
  over from 128. In a memory- or latency-bound streaming loop neither is the floor (there the fp4
  weights can beat a bf16-weight control on traffic). A route comparison against MXFP8 cannot isolate
  this — it also changes the MFMA rate.
- **Counters (rocprofv3 1.3.5, gfx950):** busy cycles / `MfmaUtil` count scaled MFMA; **mixed fp8 x fp4
  lands in `SQ_INSTS_VALU_MFMA_MOPS_F8`** (not F6F4), fp4 x fp4 in `..._MOPS_F6F4`, the upcast route in
  `..._MOPS_BF16`. Compare `MfmaUtil` only within one route. Verify:
  `rocprofv3 --kernel-include-regex <k> --pmc SQ_VALU_MFMA_BUSY_CYCLES SQ_INSTS_VALU_MFMA_MOPS_F8 -- ...`.
  If torch and rocprofv3 load two copies of `libamd_comgr.so.3` (`Option 'spirv-expand-step' registered
  more than once`), make them one file first (symlink in an image; a hard link only holds in a running
  container): `../method/profile.md ## Profiler-capability preflight`. That workaround depends on the environment.

## Verification commands (one line each)

- k_width / scale numerics + emitted op: `python3 scripts/mfma_scaled_check.py`
- spills per sweep cell: `grep -E '\.(vgpr_count|vgpr_spill_count):' <TRITON_CACHE_DIR>/**/<kernel>.amdgcn`
- assert-free register-only scale path: `gl.convert_layout(x, OUT, assert_trivial=True)` compiles
- combine cost split: `rocprofv3 --kernel-trace` on the op; compare GEMM vs reduce vs fill kernels
