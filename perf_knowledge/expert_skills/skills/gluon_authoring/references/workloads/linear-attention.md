# Chunked Linear Attention / Gated Delta Rule — prefill and decode (gfx950 / gfx942)

Gated DeltaNet, KDA (Kimi Delta Attention), delta-rule and the linear-attention family: a
recurrent state `S [D_v, D_k]` carried from chunk to chunk (prefill) or token to token (decode)
instead of a score matrix normalized within one. The attention pages do not fit it — their
variable set assumes a softmax between two matmuls, and this archetype has neither the softmax
nor the `[Br, Bc]` intermediate. The body, not the name, decides the archetype
(`index.md`, opening paragraph) before attention's variables are mapped onto it.

**Evidence base.** Triton ships no linear-attention example on 3.6.0 / 3.7.0 / 3.7.1 / 3.8.0.
The rules marked *verified* were measured on Triton 3.8 / gfx950 on a production KDA pair — a
chunked prefill (`prepare` kernel building the per-chunk WY workspace, then a `walk` kernel
carrying an fp32 128x128 state per head across 64-token chunks, plus a `scan` kernel) and a fused
single-launch decode (conv + gate + delta update of an fp32 state held in place in a paged cache +
gated RMSNorm, with a multi-token MTP/spec variant). Unmarked statements are derived, not measured.

## Measured levers (KDA pair, gfx950)

Levers that moved the measured KDA pair (one kernel family, Triton 3.8 / gfx950) where their signal
applied. Numbered for reference, not an order; each item names its condition and how to verify it.

### Chunked prefill (state carried across chunks)

1. **Fill the GPU across the chunk chain when the grid under-fills** (*verified* on the measured
   prefill): signal `N * H * (D_v / BV)` CTAs well under the CU count; the
   remedy measured was splitting the chain into `G` groups + scan
   (`## Chunk-chain split (G groups + scan): the affine prefix scan in practice`); where the grid already
   filled the device, the single walk (`G=1`) won. A/B forced `G=1` on one long and one multi-sequence case.
2. **Cross-kernel bytes moved more than per-kernel stalls** (*verified* on this prefill): sum bytes over all dispatches
   (`hw_budget.py --tensors` per kernel). Concrete cuts: chunk-major workspace (`[NT, H, 64, *]` per
   chunk), triangular `A_qk` packed to its nonzero 16x16 blocks, one fused transfer pass reading
   `w / u / kg` once per group, `o` emitted from the group-0 pass.
3. **Full-row epilogue stores** (*verified*, small). MFMA-layout stores of `w / u / A_qk`
   write 8 B per lane in 32 B row pieces; restaging through LDS into 16 B-per-lane full-row stores
   cut VMEM instructions and L2 requests in the prepare kernel. Mechanism and acceptance:
   `../tile-programming/memory-path.md ## Epilogue stores: MFMA layout vs an LDS-staged row-major store`.
4. **`nt` (`.cg`) loads paid on the read-once pass** — condition `BV == D_v` (tiles read exactly once);
   on every pass or a `BV < D_v` pass it lost in the measured prefill (*verified*, small).
5. **Walk: one-chunk-ahead register prefetch beat an async LDS ring** (*verified*, small): prefetching
   the next chunk's tiles and removing the loop-header `s_waitcnt vmcnt(0)` merges. Depth 2 and an async
   ring in this walk did not pay; the ring paid in the fused transfer pass, where it removed bytes. Re-test
   the ring if a walk issues more bytes or MFMAs per iteration than this one.

**Measured lost or tied on this KDA prefill** (*verified*, gfx950; each closes the recipe tried, re-test
when the limiter or the grid differs): occupancy raises, LDS bank-conflict cuts on the lock-stepped walk,
`llvm_fn_attrs` strategies (`## Getting the pipeline, and two ways of being misled about it`), fusing the
scan, and **overlapping prepare and walk — neither separate streams nor graph branches paid there**
(HIP graphs on ROCm 7 / gfx950 ran the captured fork/join branches serially, confirmed by a spin-kernel
microbenchmark; eager launches on two streams did overlap in that microbenchmark).

### Decode (one token, or a few MTP tokens, per sequence)

- **Placement against the in-footprint RMW floor** (*verified*) tells how much headroom is left. A T=1
   decode at large batch is an in-place fp32 state read-modify-write; within a small margin of that
   floor little is left (the measured kernel stopped there):
   `### Place decode against an in-footprint RMW probe (mem_bw_probe --rmw)`.
- **MTP / spec decode: no global load inside the token loop** (*verified*; the largest decode effect
   measured on this kernel). `vmcnt` retires loads and stores in issue order, so a load inside the loop waits
   behind the previous token's state-snapshot stores:
   `### Multi-token decode: vmcnt retires stores in order — global loads in the token loop wait behind stores`.
- **State slab cache bits: `.cg` loads + `.wt` stores were fastest on this kernel** (*verified*); `.cs` / `nt` stores
   lost badly in the kernel and the probe: `### Cache modifiers on the state slab (.cg load, .wt store)`.
- **Small batch (B1/B16): the scalar load chain and the sigmoid** (*verified*, small):
   `### Small-batch decode: the scalar (lgkmcnt) chain`.

Directions measured negative for this decode kernel are listed in `### Decode: measured negative (KDA, gfx950)`.

## The loop shape, and where the state lives

Per chunk `i`: two matmuls and one state update.

```text
intra-chunk : A_i   = Q_i @ K_i^T          (masked to the chunk)
cross-chunk : O_i  += Q_i @ S_{i-1}        (the recurrence's read)
state       : S_i   = K_i^T @ V_i + S_{i-1} * decay_i
```

That is gated linear attention. A delta rule (GDN, KDA) carries a write strength `beta_i` and
writes a **residual** `u_i = beta_i (v_i - S_{i-1} k_i)`; because `S` is updated inside the chunk,
the rows of a chunk couple. They need not be serialized: substituting the recurrence into itself turns
the chunk into one unit-lower-triangular solve,

```text
(I + T) U = beta (V - diag(p) S_in K^T),   T = tril(-beta_i (K K^T)_ij D_ij, strict)
```

`T` is *strictly* lower triangular, so `I + T` has unit diagonal and is invertible for every input
— a structural guarantee, not a conditioning assumption. Production computes the inverse once per
chunk in the prepare kernel (the WY `w`, `u` workspace) so the walk only does matmuls; on the
measured KDA prefill the prepare kernel was the largest single kernel in every case and ended at
its copy floor.

**`S` is `[D_v, D_k]`, with no chunk-length term.** A flash-attention accumulator `[Br, D]`
shrinks with the query block; `S` is the same size at chunk length 16 or 128. The measured walk
splits it by `D_v` slices (`BV`) across workgroups and keeps each slice in MFMA accumulators.

## The recurrence: fold the decay into the accumulator operand

```text
correct : S = mfma(v, k, S * decay)          # decay applied to the operand the MFMA adds
wrong   : S = S * decay + mfma(v, k, S)      # counts S twice
```

Same defect, same signature as the online-softmax rescale in
`../gluon/matrix-reference.md ## Accumulator + per-step rescale (online normalization)` — **it
passes at a single chunk and fails from two on**, because the first chunk's decay multiplies a
zero state. Shipped form (excerpt from the production KDA walk step, compiled and correct on
Triton 3.8 / gfx950; `MMA` is a version-4 `[16, 16, 32]` layout, `A_OP`/`B_OP` its bf16 dot-operand
layouts, `dec` the per-`k` decay):

```python
sb = S.to(gl.bfloat16)                                    # S as a multiplicand: bf16
sa = gl.convert_layout(sb, A_OP, assert_trivial=True)
v0 = u0.to(gl.float32) - gl.amd.cdna4.mfma(sa, w.slice(0, TH).permute([1, 0]).load(B_OP), zero)
v0 = v0.to(gl.bfloat16)
S = gl.amd.cdna4.mfma(                                    # S as the accumulator: fp32
    gl.convert_layout(v0, A_OP, assert_trivial=True),
    kg.slice(0, TH, dim=1).permute([1, 0]).load(B_OP),
    S * dec[None, :],                                     # decay folded into the C operand
)
```

## Matrix rate: S stays fp32 in the accumulator, bf16 only as a multiplicand

Keeping `S` in fp32 does **not** put the state update on the fp32 matrix path: the accumulator
of a bf16 MFMA is fp32 anyway. Only the matmuls that read `S` as an **A/B operand** (`S w^T` for
the residual, `q S^T` for the output) choose a rate, and the shipped walk casts `S` to bf16 for
those (`v_cvt_pk_bf16_f32`; its hot loop issues only `v_mfma_f32_16x16x32_bf16`, *verified* in
the ISA) while carrying the fp32 copy — and passes the RMS-ratio oracle. The rates an fp32
multiplicand would select instead (`kernel_workflow/scripts/kernel_tools/amd_isa.py --arch gfx950 facts <opcode>`):

fp32 `16x16x4` / `32x32x2`: 64 FLOP/cyc each vs bf16 `16x16x32`: 1024 (`C/D` VGPRs 4 / 16 / 4).
`32x32x2` is not faster than `16x16x4` on gfx950 (CDNA4 ISA cycle counts) — same rate, four times the
accumulator. Whether the multiplicand cast is admissible is
an oracle question first (`../tile-programming/low-precision.md`): the state feeds back into its
own next update, so a rounding introduced there compounds. The **stored** state's dtype is usually
not yours to change at all — the serving cache owns it (fp32 in the measured vLLM paged cache).

The chunk length (64) fills the MFMA M; this is not the M=1 regime of
`../gluon/matrix-reference.md ## When the matrix core is the wrong instrument (gl.dot_fma)`. The
decode step *is* that regime: the measured decode kernel is VALU-only (packed fp32
`v_pk_mul / v_pk_fma / v_pk_add` and DPP row reductions), no MFMA.

## Chunk-chain split (G groups + scan): the affine prefix scan in practice

The state update is an affine map, `S_i = A_i * S_{i-1} + B_i`, and affine maps compose
associatively:

```text
(A_2, B_2) . (A_1, B_1) = (A_2 * A_1 , A_2 * B_1 + B_2)
```

So a sequence's chunk chain can be cut into `G` groups that run in parallel. The shipped KDA
prefill does exactly this in three dispatches, and it is what filled the machine on the measured
prefill (*verified*, rank 1 above):

- **transfer pass** — one program per (sequence, head, group, slice) walks its group from a zero
  state and emits `B_g` (fp32 `[V, K]`, the group's state contribution) and `M_g` (the group's
  `[K, K]` transition). For a delta rule `A_i` is **dense** — the delta correction contributes a
  `K^T (I+T)^{-1} beta K` term no decay choice makes diagonal — so `M_g` is a matmul chain, not a
  product of scalars;
- **scan** — composes `(M_g, B_g)` serially over groups to produce every group's entry state
  (a short chain: `G` steps, not `chunks`);
- **replay pass** — each group re-walks its chunks from the true entry state and writes `o`.

The decision is **CTA count against CU count, then chain length** — not a position on a
decode-to-prefill axis. The shipped host rule (Python, from the production config):

```python
programs = num_seqs * H * (128 // BV)
g = min(split_programs_per_cu * num_cus // programs,   # fill the CUs, no more
        chunks_per_seq // split_min_group_chunks,       # groups long enough to amortize
        split_max_groups)
G = 1 if (chunks_per_seq < split_min_chunks or g < split_min_groups) else g
```

Costs, all measured: the walk work roughly doubles (transfer + replay), plus `B_g / M_g` traffic
and a scan dispatch; on multi-sequence batches that already fill the device the split lost at
every `G` tried. Wave quantization matters: once the replay pass is at one CTA per CU, the next
`G` overflows into a second partial wave. `gl.associative_scan` does not express this — it scans
tile elements along an axis, not state matrices across chunks, so it fits inside a chunk. The
cross-dispatch handoff of running state is the hazard class flagged in `index.md ## scan`.

## The Independence rule is not what blocks you here

The recurrence lives on the accumulator side, not in the prefetch buffers, so
`../tile-programming/pipeline.md ### Independence rule (correctness of pipelining)` is not
violated: each chunk's `Q / K / V / gate` tiles are independent of every other chunk's. Plain's
pipeliner also accepts a body with two chained dots and loop-carried accumulators
(`../gluon/pipeline-reference.md ## And on attention — two dots chained through a softmax`) —
evidence about legality, not a route. What the recurrence does block is **symmetric ping-pong**
(the same dependency shape as the online-softmax recurrence); buffering depth stays available.

## The state is an occupancy budget line; whether it is the limiter is a separate question

- **Budget line (arithmetic, before running).** A `[128, 128]` fp32 state on a four-wave
  workgroup is about 64 VGPRs per lane, held for the whole loop, before staging.
- **Limiter (measurement).** On the measured walk the binding term was LDS, not VGPR (*verified*): both
  divisors in `probe.py measure` show which one binds
  (`../hardware/planning-constants.md ## Occupancy is a one-sided criterion`).

If the SPI limiter reading is empty and matrix utilization is low, the fork in
`../hardware/bound-class-signals.md ## under-fill vs occupancy (the low-MfmaUtil fork)` points
the other way (grow the tile). On this workload the walk passes ran at about one WG per CU with
an under-filled grid — the fill there came from the chunk-chain split above, not occupancy.

### Which tier a state size lands in

VGPR ladder and the register-only `; Occupancy:` comment: `../hardware/planning-constants.md ## VGPR / occupancy thresholds (CDNA combined budget)`
and `../hardware/planning-constants.md ## The emitted ; Occupancy: N is a register-term answer`,
`kernel_workflow/scripts/kernel_tools/amd_occupancy.py`; then convert to workgroups/CU
(`../hardware/planning-constants.md ## waves/SIMD is not workgroups/CU — convert before spending a round on it`).

## Getting the pipeline, and two ways of being misled about it

- **`num_stages=` on a Gluon launch is inert** on Triton 3.8.0 / gfx950 (*verified*: the walk kernel
  compiled with `num_stages=4` is ISA-identical to the default; a compile-only loop check agrees). Depth comes from hand-authored staging — the
  measured walk carries its own `NUM_STAGES` constexpr for a one-chunk-ahead register prefetch.
  A null result from turning the launch knob up is not evidence that the loop resists pipelining.
  Order of the authored rungs: `../tile-programming/pipeline.md`; why the knob is dropped:
  `../gluon/pipeline-reference.md ## Roll the loop (cut i-cache pressure)`.
- **`llvm_fn_attrs` scheduler strategy on the measured chunk recurrence: neutral to negative**
  (*verified*); **`amdgpu-sched-strategy=iterative-ilp` segfaulted the compiler** (exit 139, no Python
  traceback) on the single-walk pass and on the T=1 decode kernel (a small loop compiles, so the crash
  is kernel-dependent). Mechanism: `../tile-programming/llvm-fn-attrs.md`.
  `loop_unroll_factor` is a loop keyword (`tl.range(..., loop_unroll_factor=U)`), not a launch
  keyword; a chunk loop whose bound is loaded from `cu_seqlens` has no `gl.static_range` form.
- **Re-injecting plain's pipeliner** (a diagnostic; it is not a route for an incumbent Gluon kernel):
  `buffer_ops=True` conflicts with a body that already uses `gl.amd.cdna4.buffer_load` /
  `buffer_store`, including in early-exit branches; it fails legalization. Rule and failure
  strings: `scripts/gluon_swp.py`.

## What the language gives you, on all four versions

| Need | API | Note |
| --- | --- | --- |
| the chunk loop | a plain Python `range` | Gluon exports `static_range` and no `range` (3.8.0 checked); a plain `range` lowers to `scf.for` with `S` as an `iter_args` value |
| state seed | `gl.zeros(shape, dtype, layout=...)` | pass the layout explicitly |
| within-chunk decay accumulation | `gl.associative_scan` | `tl_core.associative_scan` re-exported — **no `layout=` kwarg**; Gluon exports no `cumsum` |
| chunk matmuls and the state update | `gl.amd.cdna4.mfma` on gfx950 (`gl.amd.cdna3.mfma` on gfx942) | `S` is the accumulator operand, per the fold above |

`gl.dot_fma` accepts a rank-3 accumulator only from 3.8.0, which also warns when the joined
`M x N x K` (times batch) exceeds 2^19; `loop_unroll_factor` takes effect from 3.8.0.

## Decode: the in-place state RMW stream

One token per sequence (or `T` MTP tokens): load the 64 KB fp32 state slab per (sequence, head),
update it, store it back in place, plus a few KB of conv state and inputs. Bytes are dominated by
the state, and the decode kernel is spill-free at several waves/SIMD (check with `probe.py measure`). `hw_budget.py`
routes it to `bound: memory` correctly — declare it with `--tensors` (state read and write as
two entries) and `--flops-engine valu`, which reports "no VALU peak tabulated: pass --measured-tflops" because no VALU
peak is tabulated; that caveat does not change the routing.

### Place decode against an in-footprint RMW probe (mem_bw_probe --rmw)

**The in-footprint probe is `mem_bw_probe.py --rmw --footprint-mb <state working set>`** (in-place read-modify-write of the
kernel's own footprint, graph replay + read-evict through the task harness timer, `--eager` to match
an eager timer; allowed below the MALL bar, the tool requires a footprint of at least twice L2) and its printed `--measured-hbm-tb-s`
feeds `hw_budget.py`. Why not the default mode: it measures separate read and write regions at a
footprint of at least 4x the 256 MiB MALL and refuses anything smaller ("footprint ... is under 4x the
256 MiB memory-side LLC"). A decode state of 50-400 MB is below that, and its in-place RMW ran **faster than the
probe's ceiling** (*verified*): feeding the probe's `--measured-hbm-tb-s` into `hw_budget.py`
put the floor above the measured kernel and `hw_budget.py` exited 2 ("measured time is BELOW the
floor"). Exit 2 here means the ceiling is too low, not that the inputs are wrong. On a tool build without
`--rmw`, the same probe by hand (compiled and run on Triton 3.8 / gfx950; time it with the task
harness's graph-replay + read-evict timer so both sides see the same cache state):

```python
@triton.jit
def rmw(a, SLAB: tl.constexpr, BLOCK: tl.constexpr):     # one program per (seq, head) slab
    base = tl.program_id(0).to(tl.int64) * SLAB
    for i in tl.static_range(0, SLAB, BLOCK):
        off = base + i + tl.arange(0, BLOCK)
        x = tl.load(a + off, cache_modifier=".cg")
        tl.store(a + off, x * 0.999 + 1e-3, cache_modifier=".wt")

a = torch.randn(N * H * 128 * 128, device="cuda") * 0.1   # realistic, non-zero data
rmw[(N * H,)](a, SLAB=128 * 128, BLOCK=4096, num_warps=4)
```

Observed on gfx950 with this probe: the achieved rate depends on the footprint (lower for a
small batch's short stream, lower again past several GB), in-place and read-A/write-B tie, and
random vs zero data tie. The decode kernel at large batch landed within a small margin of the
probe plus its conv-state traffic — where the measured work stopped.

### Multi-token decode: vmcnt retires stores in order — global loads in the token loop wait behind stores

`vmcnt` counts loads and stores in issue order, so a load inside the MTP token loop drains the
previous token's snapshot stores. What removed it (*verified*, the largest decode gain on the measured
kernel): the slot indices loaded up front and every token's inputs (q/k/v/g rows, out-gate, beta) staged
into LDS before the token loop, so the loop issues no global loads. The ISA then shows whether any
`s_waitcnt vmcnt` is left inside the loop — two traps:

- `gl.amd.cdna4.async_copy.wait_group(0)` lowers to `wait_asyncmark(0)`, which becomes
  `s_waitcnt vmcnt(k)` with k = VMEM ops issued **after** the last async copy — not a `vmcnt(0)`
  drain. Ordinary loads issued before that copy are settled (in-order counter); a loop-carried value
  loaded after it is not.
  Correctness is never at risk (LLVM waits before the first use); the question is only *where* that wait
  lands. In the simple store-heavy loops GPU-timed on gfx950 / 3.8.0, LLVM already put it in the loop
  preheader and the settle below tied — read the loop ISA first and add the settle only if the first-use
  `vmcnt` sits inside the loop.
- A non-pure identity asm pins such a value; LLVM's waitcnt pass then waits once at that
  point, outside the loop (shipped, compiled and correct on Triton 3.8 / gfx950):

```python
@gluon.jit
def _settle(x):
    y = gl.inline_asm_elementwise(
        "v_mov_b32 $0, $1", "=v,v", [x.to(gl.float32)], gl.float32, False, 1
    )
    return y.to(x.dtype)
```

Interleaving each token's snapshot `buffer_store_dwordx4`s with the update added a small gain (*verified*).
On Triton 3.8.0 / gfx950 `gl.store(..., cache_modifier=".wt")` emits a trailing `s_waitcnt vmcnt(0)`
after each store, while `gl.amd.cdna4.buffer_store(..., cache=".wt")` sets the same bits without the wait
(compile-checked). GPU-timed on gfx950 / 3.8.0: the per-store drain cost heavily when few waves per CU run a
store-heavy loop and was hidden at full occupancy: at low occupancy the buffer form avoided the drain; at
full occupancy the two tied. After the fix the MTP case sat close to
the measured gfx950 write ceiling for its snapshot bytes; earlier `num_warps` and store-bit wins on
the unfixed kernel were the same drain stall and disappeared.

### Cache modifiers on the state slab (.cg load, .wt store)

On gfx950 / gfx942, Triton 3.8, **buffer ops**: load `.cg` and `.cs` lower to `sc0 nt`, `.cv` to
`sc0 sc1`; store `.wt` lowers to `sc0 sc1`, `.cs` to `sc0 nt`, `.cg` / `.wb` set no bits. **Global ops**
(`gl.load` / `gl.store`) drop `sc0`: load `.cg` / `.cs` -> `nt`, store `.cs` -> `nt`; `.cv` load / `.wt`
store -> `sc0 sc1` (and `gl.store .wt` adds `s_waitcnt vmcnt(0)` after each store). For the in-place state RMW,
`.cg` loads with `.wt` stores were fastest, and `nt` stores (`.cs`) behind `.cg` loads lost by the
largest margin of any combination — both in the production kernel and in the probe above (*verified*,
gfx950, this state-slab RMW). A footprint sweep (gfx950 / 3.8.0, Gluon buffer ops, tens of MB to GB-scale slabs)
kept `.cg` / `.wt` best or tied at every size; the `.cs` loss was largest while the slab fits the
memory-side cache and shrank to a tie with plain stores at GB scale. The measured kernel's form:

```python
s = gl.amd.cdna4.buffer_load(s_src + i * K, off_s, cache=".cg").to(gl.float32)
...
gl.amd.cdna4.buffer_store(S.to(s_dst.dtype.element_ty), s_dst + i * K, off_s,
                          mask=s_keep, cache=".wt")
```

The measured kernel keeps state loads and stores at 16 B per lane (`dwordx4`); a masked buffer store can break
immediate-offset folding — an out-of-range offset (`where(valid, 0, 0x20000000)`) keeps
`dwordx4` and drops the write.

### Small-batch decode: the scalar (lgkmcnt) chain

At small batch the time is a chain of dependent scalar loads (kernarg → slot index → kernarg → state
address). What shortened it was collapsing it into one `lgkmcnt` hop: every slot-independent load first, then one burst of vector loads
(inputs, conv taps, state slab), then the consumers (small, *verified*). Tools:

- a side-effecting empty asm on the uniform value keeps it in an SGPR and stops LLVM from hoisting
  a dependent `s_load` or branching on it — `../tile-programming/instruction-scheduling.md ## The empty-asm scheduling fence — the most-used and least-documented lever here`;
- put every slot `s_load` **before** the first side-effecting asm: a uniform load after it is no
  longer provably unclobbered, becomes a VMEM load, and the `=s` constraint then fails or faults;
- a libdevice `tanh`-based sigmoid has two-sided divergent branches; `exp2` + `rcp` was a small gain
  where the tolerance allowed it.

Judge these with the harness's flushed graph-replay timer and interleaved A/B; a warm trace
median hides them. An XCD remap of the program id helped the largest batch and cost the smallest;
grid size was the deciding signal.

### Decode: measured negative (KDA, gfx950)

Measured lost or tied on the gfx950 KDA decode kernel (*verified*; each closes the recipe tried — re-test
if batch, state size or the VGPR budget differ): **split-V across workgroups with a cross-CTA gated-RMSNorm
reduction** (the extra memory round trips exceed the small-batch body time; at MTP it does not
shorten the per-token chain); persistent / grid-stride schedules (every T=1 workgroup was already
co-resident); forcing `waves_per_eu` (spills); `amdgpu-sched-strategy=iterative-ilp` (compiler
segfault); `gl.atomic_add(sem="acq_rel", scope="gpu")` in a hot decode kernel — it lowers to
`buffer_wbl2 sc1` / `buffer_inv sc1` (Triton 3.8.0, compile-checked), an L2 writeback; `relaxed` emits
no maintenance when no reader in the launch needs it
(`collective.md ## The one primitive: an atomic carrying sem and scope`).

## Variable-length sequences

When the chunk count is data-dependent, the three consequences in
`attention.md ### The dynamic-body variable (sparse / paged / varlen)` apply unchanged: the index
chain is VALU work to be placed, the remainder path is a different region structure, and the
budget is a distribution. Read the acceptance signal on the spread of worktile durations. For the
chunk-chain split, decide per call from `num_seqs`, `H` and chunks per sequence (the host rule
above) — one varlen batch and one long sequence want opposite answers.

## Mirror risk: staying in fp32 where the oracle rounded

The risk on this page is losing precision you needed. The opposite — keeping fp32 across a point
where the reference materialized bf16, so the fused kernel is *more* accurate than the oracle and
fails a tight one — is in `reduction-elementwise.md ## Stage 3 — dequant + residual + norm`.
