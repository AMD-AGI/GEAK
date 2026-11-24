# Classes 4 and 4b: synchronization, and protocol blocks

Part of `../inline-asm-reference.md`. Classes 1–3: `classes.md`; 5/6: `classes-lane-register.md`.

## Class 4 — synchronization

A wait or publish whose granularity the binding does not offer. **Risk:** this class removes the
compiler's dependency checking — a hand-written `s_waitcnt` that is wrong is a wrong answer at a shape you
have not run. The binding is not a full safety net either: on 3.8.0 a missing `commit_group` or a
wrong `wait_group` count also compiles into a silent under-wait (`../pipeline/async-ordering.md ## Getting wait_group right`).

### The drain marker — a different obligation from `wait_group`

```python
gl.inline_asm_elementwise("s_waitcnt lgkmcnt(0)\n v_mov_b32 $0, 0", "=v,~{memory}", [],
                          dtype=gl.int32, is_pure=False, pack=1)   # result deliberately discarded
```

**`wait_group` retires the producer (copy completion); this retires the LDS readers**, so the next
`buffer_load_to_shared` cannot overwrite a slot still being read. Removing it leaves a race: on 3.8.0 /
gfx950 the refill then issues straight after the `ds_read` with no `lgkmcnt` wait (compile-only check). Place it
between the last relaxed LDS read and the next copy: `wait_group` → `barrier` → relaxed loads → **drain**
→ `barrier` (a one-wave region may elide a barrier — say so in a comment).

### `s_waitcnt vmcnt(N)` — staged partial drains

A **pacing** lever, not a readiness lever; trade-offs in
`../../tile-programming/pipeline.md ### Draining below the group: bare s_waitcnt instead of wait_group`.
A `commit_group` placement may already give the granularity and keeps the compiler's checking (`wait_group` counts groups, not copies —
`../pipeline/async-ordering.md ## Getting wait_group right`).

- **Depth as an operand:** `"s_waitcnt vmcnt($1)"`, `constraints="=v,n,~{memory}"` — `n` forces a
  compile-time immediate. A `vmcnt(<digits>)` text audit is blind to this form.
- **The count is the per-panel VMEM instruction count — derive and assert it:**

```python
# each wave issues one 128-bit A copy and one 128-bit B copy per panel
gl.static_assert((0 <= NEWER_PANELS) & (NEWER_PANELS <= MAX_PANELS))   # no chained compares in @jit
gl.inline_asm_elementwise("s_waitcnt vmcnt($1)", "=v,n,~{memory}", [2 * NEWER_PANELS],
                          dtype=gl.int32, is_pure=False, pack=1)
```

- **Derive and assert the count from the per-panel copy count, and confirm the issue order in the `.s`**
  (the M0 sequence of `buffer_load ... lds`). A class-2 empty-asm marker between copy groups is cheap
  insurance, not what keeps a correct count correct: on Triton 3.8.0 / gfx950 a side-effecting
  (`is_pure=False`) wait asm already kept the LDS-DMA copy issues in program order, with or without the
  marker (GPU stale-read test with a deliberately wrong count as positive control). Validate a new count
  the same way.

### The wait's operand list can be the entire ordering statement

The block's operands are its dependencies (payload anchors; an input with no output slot is a pure
anchor) — deleting "unused arguments" silently drops the ordering.

### The dependency-carried drain — a wait with no `s_waitcnt` in it

Tie the result of an LDS read through an empty asm with `~{memory}`: the backend inserts the
`lgkmcnt` wait before it, and the clobber holds the next async copy after it (M0 site, intent I-B;
compile-checked on 3.8.0 / gfx950 at `num_warps=1`).
Preconditions: the slot is **wave-private**; the tied result **replaces** the original downstream; the tie
covers the **whole** read (slot count per the arity law, `pack` derived from the read — a wrong `pack`
compiles and splits or pads the tie, `classes-lane-register.md`).
Keep the explicit `s_waitcnt lgkmcnt(0)` on the other arm of a constexpr. Worked derivation:
`costs.md ## Worked derivation — a shared-memory reader drain at a new shape`.

### `buffer_wbl2` / `buffer_inv` — release and acquire

The bound route: a `gl.atomic_*` flag with `sem=`/`scope=` emits these; on **uncached** peer memory neither
is needed (`../../workloads/collective.md ## The other route: uncached peer memory, plain flag stores`).
For a publish the compiler does not see as a release into cacheable memory: agent `buffer_wbl2 sc1` /
system `buffer_wbl2 sc0 sc1` + `s_waitcnt vmcnt(0)` (plus `lgkmcnt(0)` when LDS / scalar ops are outstanding;
LLVM's own fences, gfx950 = gfx942); acquire after a successful poll: system
`buffer_inv sc0 sc1`, agent `buffer_inv sc1` (or read the payload with `.cv` loads instead). **The exemption belongs to `gl.atomic_*` with
the right `sem=`/`scope=`**, not to "an atomic was involved" — a hand-written atomic, or `scope="gpu"`
toward a peer device, gets nothing emitted (`../../workloads/collective.md ## Publish and consume`). **A
*terminal* wait with no payload reads after it correctly omits `buffer_inv`** — do not "fix" the asymmetry.

### The publish fence — drain, barrier, opaque zero

```python
z = gl.inline_asm_elementwise("s_waitcnt vmcnt(0)\ns_barrier\ns_mov_b32 $0, 0", "=s,s,~{memory}", [x],
                              dtype=gl.int32, is_pure=False, pack=1)
gl.store(flag_ptr + z, gen, mask=lane == 0)        # the zero in the ADDRESS orders the flag after the fence
```

Every wave retires its own stores, then the CTA barrier, then one lane writes the flag. The `"s"` input is
only a carrier (the body never reads `$1`): with a per-lane `x` (e.g. from `gl.arange`) LLVM inserts a
`v_readfirstlane_b32` and prints no illegal-copy diagnostic (compile-only, 3.8.0, gfx950 and gfx942), unlike
the `"=s,0"` tie of `classes-lane-register.md`, whose result is the input. No binding on
3.8.0: `gl.barrier()` emits no store wait (gfx950 compile-only check), and a release atomic is understood to drain only its own wave (unverified at run time: a cross-XCD gfx950 test saw no stale payload without the per-wave drain). A wave-only form
(`s_waitcnt vmcnt(0)` without `s_barrier`) serves a flag published per wave. On cacheable memory put
`buffer_wbl2 sc0 sc1` before the drain. Gate it in the ISA: on every path after the fence the first
store is the flag. The snippet's plain `gl.store` flag is the uncached-memory form. On cacheable memory
(ordinary `hipMalloc`) write the flag itself as a `.wt` store (`gl.store(..., cache_modifier=".wt")` /
`buffer_store(..., cache=".wt")`) or as a release-`sys` atomic: a plain flag store on cacheable memory broke a
two-GPU protocol (gfx950 / Triton 3.8.0, `../../workloads/collective.md ## Rendezvous invariants`), and
`buffer_wbl2` followed by a plain flag store was not tested. API-only fallback: the `release`/`sys` atomic as
the flag.

### A gate on the value cannot see a missing acquire

Value gates (repeats, oracle, race-test) are no evidence for memory ordering: argue the release/acquire
pairing and scope, or read it in the ISA; label a relaxed-ordering speedup unverified otherwise. Still run the race-test
(`../../method/benchmark-hygiene.md ## Determinism race-test (async / barrier / pipeline / layout changes)`) — it catches wrong counts, not missing fences.

### Half a pair guarantees nothing

Many waits are the completion half of a split issue/complete pair linked only by a tie — record and audit
`PAIRED-WITH` (`classifying.md ## PAIRED-WITH`).

### `s_and_saveexec_b64` — not a `gl.warp_predicate` substitute

`warp_predicate` (fork-only) lowers to `s_and_saveexec_b64` + `s_cbranch_execz`, but the asm door takes
and returns tensors, so narrowing EXEC skips only the block's own instructions — **the 3.8.0 elementwise door cannot skip
Gluon-emitted code** (language ceiling, `../../hardware/capability-matrix.md`). You *can* write a
role-split whose branches are entirely asm text (Class 4b).

## Class 4b — protocol blocks

**Mechanical test:** body contains `s_cbranch*`, a numeric label, `${:uid}`, or `s_*saveexec_b64`. A
*program*, typical of multi-GPU collectives; removing it is a data race. Canonical acquire:

```python
gl.inline_asm_elementwise("""
    v_cmp_gt_u32_e32 vcc, 64, $3          ; only wave 0 participates
    s_cbranch_vccz .Ldone${:uid}          ; skip the WHOLE body otherwise
  .Lpoll${:uid}:
    global_load_dword $0, $1, off sc0 sc1 ; system-coherent, or the poll can keep reading a stale line
    s_waitcnt vmcnt(0)
    v_subrev_u32_e32 $0, $2, $0           ; signed difference: wrap-tolerant
    v_cmp_gt_i32_e32 vcc, 0, $0
    s_cbranch_vccnz .Lpoll${:uid}
    buffer_inv sc0 sc1                    ; the ACQUIRE
  .Ldone${:uid}:
""", constraints="=&v,v,v,v,~{vcc},~{memory}", args=[ptr, target, pred],
    dtype=gl.uint32, is_pure=False, pack=1)
```

Release: on uncached peer memory the publish fence above + a plain flag store; on cacheable memory with
an asm flag, `buffer_wbl2 sc0 sc1` + drain + the flag written as a `.wt`-equivalent store (`sc0 sc1`) or a
release-`sys` atomic, usually EXEC-masked (a plain flag store after `buffer_wbl2` was not tested; plain flags
on cacheable memory broke the two-GPU test, gfx950 / Triton 3.8.0). Before writing the poll as asm, check the API form: `.cv` loads (the same `sc0 sc1`)
in a `tl.condition(..., disable_licm=True)` loop (`../../workloads/collective.md ## Spinning, and the hoist that will eat your wait`).
Compiled on 3.8.0 for gfx942 and gfx950, a `.cv` poll stays inside the loop as `global_load … sc0 sc1`
(also without `disable_licm`); a plain or `volatile=True` load of a uniform address becomes one
`s_load_dword` hoisted above the loop even with `disable_licm=True` — a spin that never re-reads. An
`acquire`/`sys` atomic poll compiles to `global_load … sc0 sc1` + `buffer_inv sc0 sc1` per iteration.
What the asm loop adds: all sources polled per iteration under one wait, and the wall-clock bound.

### Seven preconditions, every one load-bearing

1. Lane/wave predication is an **operand** built under an explicit layout — `gl.arange(0, 256, layout=gl.BlockedLayout([1], [64], [4], [0]))`, elect with `v_cmp_lt_u32 vcc, $2, 64`.
2. Branch over the **entire** body for non-participants (it touches peer memory).
3. `sc0 sc1` on the poll load (gfx942 / gfx950 spelling) — the bits `cache_modifier=".cv"` emits on `gl.load` and `buffer_load`; `.cg` / `.cs` / `volatile=True` do not.
4. Compare as a **signed difference** (`v_subrev_u32` then `v_cmp_*_i32 …, 0`); unsigned compare deadlocks on rollover.
   The signed form passes at once on a never-written (zero) flag once the target exceeds 2^31; where a
   flag can only hold this call's value or an older one, use a never-zero generation and an equality
   compare (`../../workloads/collective.md ## Rendezvous invariants` 3).
5. EXEC saved and restored on **every** exit: `s_and_saveexec_b64 $0, vcc` … `s_or_saveexec_b64 $0, $0`, `$0` an `=&s` save slot (not data); `s_cbranch_execz` targets the label *before* the restore.
6. Declare `~{scc}`, `~{vcc}`, `~{memory}`; `=&` on every output written before all inputs are read.
7. Labels must not retarget: `${:uid}` labels make it impossible. Numeric `1:`/`1b`/`2f` are safe only
   when the block defines every label it references — a `2f` without a `2:` in the same block binds
   silently to the next inlined block's label. `${:uid}` compiles to the same machine code; if you keep
   numeric labels, check self-containment when the string is built.

### Spin blocks built on the host, and bounded

Protocol strings that vary with world size or source count are best generated on the host as a checked
product and passed as `constexpr` (`asm`, `constraints`, `nout`, `nin`): assert at generation that the
`=` count equals `nout`, every `$k` is in range and every input referenced, register classes match the
mnemonics (`s_*` only `s` slots, `v_*`/`global_*` only `v`), the counter is `=&s`, every output `=&`,
`~{scc}`/`~{vcc}`/`~{memory}` and any physical registers are clobbered, every poll load carries
`sc0 sc1`, labels are self-contained. On the device: `gl.static_assert(len(ptrs) == NIN)` and
`dtype=(gl.int32,) * NOUT`. The string is part of the specialization key — compile every variant before
the first collective call, with the same parameters on every rank.

- **Wall-clock bound:** every few thousand failed polls read `s_memrealtime` into reserved SGPRs
  (clobber them), set a deadline on the first read, on expiry set the counter to a reserved value and
  leave (optionally `s_trap`); the kernel reports, the host decides. No `gl.*` clock exists on the 3.8.0 wheel.
- **Backoff:** `s_sleep 1` per failed poll after a threshold — optional; it does not shorten a
  dependency chain.
- **Drain before the spin:** a store still in flight is waited for by the poll's own `s_waitcnt
  vmcnt(0)` (in-order retirement); drain first and gate "no store outstanding at a spin" in the ISA.
- **Flag stores the spin reads:** on cacheable memory every flag store, on both sides, is `sc0 sc1`
  (`.wt`) or a release-`sys` store — keeping locally written words out of polled lines is not enough
  (plain stores failed on separate lines too, gfx950 / 3.8.0), and a `.wt` flag is not a release for the
  payload. On uncached memory plain stores worked; one writer per line stays a precaution there
  (`../../workloads/collective.md ## Rendezvous invariants` 1).

Also seen: assembler-level `.if $8 == 0 … .endif` on an `n` operand (two bodies, no runtime branch);
raw buffer descriptors in named SGPRs (decode the word against the buffer-resource spec, never copy it).
**Verify:** cross-rank race-test + ISA read that labels, EXEC bracket and cache maintenance survived.
Hard-codes world size, record layout, wave size, sometimes physical registers — invisible to any type system.
