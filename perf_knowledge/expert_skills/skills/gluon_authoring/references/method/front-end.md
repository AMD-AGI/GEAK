# Front end: gfx950 notes for the plain-Triton tier

Only the plain-Triton facts on gfx950 / Triton 3.8.0 a strong agent would get wrong. Generic tuning
(coalescing, hoisting, autotune configs, grid order, split-K) is assumed known. `$KT` =
`kernel_workflow/scripts/kernel_tools`.

## num_stages on plain gfx950: A/B 2 vs 3

**Sweep `num_stages` together with `BLOCK_K`; the gate is the LDS budget, not a `BLOCK_K` threshold:**
`(BM + BN) × BK × bytes × num_stages` must fit the 160 KiB per CU on gfx950, and the workgroups per CU it
leaves is part of the trade. Measured on gfx950 / Triton 3.8.0 (plain bf16 `tl.dot`): `ns=3` was often the
best point — including at `BLOCK_K=32` on 256x256 tiles, where `BLOCK_K=64` at 3 stages does not fit — and
won most on a long-K, under-filled grid; it lost where its LDS dropped a workgroup per CU on a full grid;
decode tiles were insensitive. **Read the spill count per point:** the backend spilled at `ns=2` on a wide
tile where `ns=3` did not, and a recorded wide tile was clearly slower at ns=3 than at ns=1, pinned at 256
VGPRs (`recover.md ### 2.1 The plain@ns=1 control`) — so neither depth is the safe default.
Read `spill=` from `$KT/probe.py measure --dir <ir>` for each depth. The depth may be set by `tl.range(..., num_stages=N)` on the loop, which overrides
the launch argument (`recover.md`). On the Gluon path no 3.8.0 pass consumes `num_stages`.

## Confirm the plain pipeliner fired

**The pipeliner declines silently under LDS or register pressure.** Check the dump, not the knob:

```bash
bash "$KT/dump_ir.sh" python bench.py --variant plain --out ir/ --arch gfx950
grep -c 'ttg.memdesc_index' ir/plain/plain.ttgir   # >0: multi-buffered local_alloc = fired
grep -c 'async_copy'        ir/plain/plain.ttgir   # gfx950: async copy is the default path; gfx942: expect 0
```

No `memdesc_index` at `num_stages > 1` (and on gfx950 no `async_copy`) = plain did not pipeline this
loop at this config; that is a signal for an authored Gluon pipeline (`../tile-programming/pipeline.md ## Plain
Triton: the pipeliner runs for you`).

## kpack

**`kpack` is deprecated on gfx950** (`kpack is deprecated starting from gfx950`, checked on 3.7.0; not
re-checked on 3.8.0): set `kpack=1` or drop it. Test hint removal (`waves_per_eu`,
`matrix_instr_nonkdim`, `cache_modifier`) as its own hypothesis. On plain Triton the MFMA shape is part of that register budget: on gfx950 / Triton 3.8.0 the default
`matrix_instr_nonkdim=0` picks `32x32x16` for `BLOCK_M >= 32`, and a 256x256x64 bf16 GEMM tile at 8 warps
spilled under it but compiled spill-free with `matrix_instr_nonkdim=16` (`16x16x32`). Read the spill count
with both `nonkdim` values before attributing an 8-warp spill to `num_warps`. A champion's `waves_per_eu` copied
onto a Gluon anchor caps its occupancy (`transcribe.md ## 3. Apply — the step that carries the port`).

## exp2 vs tl.exp

**On gfx950 / Triton 3.8.0 `exp2` lowers to one `v_exp_f32` (compile-checked); `tl.exp` adds range-scaling ops
(`fma`/`v_cndmask`/`v_ldexp`).** Folding `log2(e)` into the existing scale constant keeps it at one `v_exp_f32`:
`p = tl.math.exp2(qk * (scale * 1.44269504))`. Partial FMA emulation when the exp unit binds:
`profile.md ### Raising exp throughput by partial FMA emulation (when the exp unit is the bound)`.

## loop_unroll_factor is live on Gluon 3.8.0

`tl.range(..., loop_unroll_factor=1)` keeps a loop rolled (i-cache pressure); unlike `num_stages` it is
honored on the Gluon path from 3.8.0. Semantics: `../gluon/pipeline/loop-knobs-and-targets.md`.

## Host side

- **`tensor.item()` / `max().item()` in a sub-ms dispatch path forces a GPU→CPU sync** (a stream
  drain plus a host round trip) — on a sub-ms bucket that can erase a kernel win; time it on the target host. Host-known values (shapes, args, dtype flags) avoid it.
- **Launcher-object escape:** a module may export an object with `__getitem__(grid) -> callable(**kwargs)`
  that owns grid, workspace, config and per-shape dispatch. Forward `**kwargs` completely, guard each
  hard-coded assumption with a fallback to the unmodified path, and **make the memo cache key cover
  every independent variable**: a plan keyed on M alone, carrying a grid from N and `EVEN_K` from K,
  produced 4.24M wrong elements of 4.33M and a memory fault on the second shape.
- **In-kernel granularity fusion:** when the caller owns the grid, one program can cover `G` of the
  caller's blocks at a `[G*BM, BN]` tile and retire the surplus CTAs in the prologue (a large measured gain on one MoE kernel; A/B it per shape).
- A dispatcher that silently falls back to the plain kernel produces a "Gluon" number that is plain's;
  check the launched kernel name.
- `fp_downcast_rounding="rtz"` on the epilogue cast is cheaper on some paths but is a numerics change
  (`../tile-programming/low-precision.md`).
