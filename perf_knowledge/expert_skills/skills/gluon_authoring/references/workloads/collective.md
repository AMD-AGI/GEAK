# Multi-GPU From Inside The Kernel (gfx950 / gfx942)

For communication fused *into* a kernel (allreduce-norm-quant, a decode step reading a peer's KV
shard). A stand-alone collective is usually served by RCCL; a custom library competes with it when it is
tuned for the case (the DDP section below records one that did). The atomic primitives
are plain-Triton builtins re-exported into `gl`. Evidence: the atomic sections are source-read against
3.8.0 (3.6/3.7 differences noted) plus one disassembly of a `gpu`-scope `acq_rel` atomic on gfx950; the
uncached-memory transport, the protocol forms and the spin bound are taken from a production all-Gluon
8-GPU collective library (all-reduce, all-gather, reduce-scatter, all-to-all; one source for 3.6-3.8,
run on gfx942 / gfx950). The design rules in `## Measured design choices (gfx942 / gfx950)` come from that
library's same-session A/Bs: inference TP all-reduce on gfx942 (MI300X / MI325X / MI308X) and gfx950, and
an fp32 data-parallel gradient all-reduce on MI325X (gfx942) and MI350X (gfx950); no effect sizes are
carried. Cache-bit and poll-loop facts are compile-only checks on 3.8.0 for both arches. gfx942 is
identical unless stated.

**No percent-of-peak here.** `perf_knowledge/hardware/data/sku.json` / `hw_constants.json` carry no
peer bandwidth or hop latency; `arch.*.l2_fabric_latency_high_cyc` is device-internal, not a peer hop.
Comparisons here are same-session A/Bs (same process, buffers, world size); one-shot vs reduce-scatter+gather is a
`TP/2` byte ratio, crossover keyed on message size (not on whether a peer workspace was passed).

## TP all-reduce: fusion, and when stream overlap has nothing to hide behind

**Overrides KB** for this scope: `perf_knowledge/operators/allreduce/fusion.md` recommends overlapping the
all-reduce with the next GEMM via streams; that needs independent work, which the inference TP path does not
have (measured on gfx950 / gfx942). Scope: the inference TP all-reduce whose output the next op consumes (residual add, norm, the next
GEMM). Nothing independent is left to hide it behind, so stream-level "overlap with the next GEMM"
(high-priority comm stream, more hardware queues) has no work to overlap unless the graph changes:
the `reduce_scatter` + `all_gather` rewrite fused with the adjacent GEMMs, or two micro-batches
interleaved. Production fuses the reduce with residual add + RMSNorm (+ fp8 quant) instead; in the
measured cases what moved this path was fewer bytes and fusion. The gradient all-reduce of data-parallel training is the opposite case:
`## Data-parallel gradient all-reduce: overlap is given, coexistence decides the step time`.

## Data-parallel gradient all-reduce: overlap is given, coexistence decides the step time

In DDP each bucket's all-reduce runs beside the rest of the backward by construction. What decides
the step time is how much the collective slows the kernels it overlaps, not its stand-alone bandwidth:
on gfx950 a collective that lost to the vendor library stand-alone at small sizes and led only narrowly
at bucket sizes won clearly end to end (and on gfx942 too, by less). The metric that tracked end-to-end
results there was the overlapped step time, or the slowdown of the overlapped kernels plus the collective's
bandwidth under overlap; stand-alone bandwidth did not.

- **Power cap and contention are separable.** Under a socket power cap the
  overlapped GEMMs slow down next to any collective (the data movement spends the power budget); lock
  the clock low enough that the cap does not bind and re-measure — the residue is real co-residency
  contention. On gfx950 the residue for GEMMs next to a CU-light collective was near zero at every
  reducer geometry, while a library holding fixed CU channels kept a large one (gfx942 at its clock
  lock: the same ordering); in those measurements only fewer bytes or less energy per byte moved the
  power-cap share, and locked kernel-time ratios isolated the contention that unlocked ones hid. Capping the vendor library's channels to leave CUs to
  compute hurt end to end on gfx950 and was neutral on gfx942 — not a free coexistence lever there.
- **Attention backward was the kernel that suffered** (high VGPR / LDS occupancy leaves few free wave
  slots): it kept a real slowdown under every reducer geometry tried on gfx950, so a reducer tuned only
  against the GEMMs can miss it.
- **Measured on these GEMM-overlap cases: few small reducer CTAs that co-reside, no CU reservation, came out ahead.** Bulk transfer on the copy engines;
  CTAs only reduce — about one word per lane per source, rank order, a dozen VGPRs, no LDS, fewer CTAs
  than CUs. A library that reserves whole CUs for its channels evicts GEMM workgroups, and the eviction
  costs more on the device with fewer CUs (gfx950's 256 vs gfx942's 304). Two waves per reducer CTA beat
  one on gfx950 (less attention slowdown, more bandwidth under overlap); more waves, more chunks per CTA,
  or pinning to a CU subset only traded attention against GEMM or bandwidth. On gfx942 (64 KiB LDS, one
  GEMM workgroup per CU) the two-wave form was not a win — re-sweep per arch.
- **A CU-masked stream bought no interference reduction** in the gfx942 A/B, and one created with
  `hipExtStreamCreateWithCUMask` is blocking with respect to the null stream. A CU mask belongs to the HIP
  queue: it is lost inside a captured graph, and a mask that leaves an XCC with no CU can hang the queue
  (production library on gfx942 / gfx950 refuses such masks; not reproduced here).
- **Buckets: large, registered in place, came out ahead in the sweeps.** Around 1 GiB: on gfx950
  every smaller bucket tried lost, on gfx942 none beat it beyond noise. Smaller buckets did not buy
  overlap there (each burst holds its CTAs longer and per-call costs add up). Registering the bucket
  itself (`gradient_as_bucket_view`, zero copy) beat a copy-in / copy-out path clearly, and handing
  the bucket to another library lost more. Sizes past the IPC window: `## Measured design choices (gfx942 / gfx950)`.

## Measured design choices (gfx942 / gfx950)

Decision rules from same-session A/Bs; each holds under its precondition only. Re-measure on a new SKU:
the crossovers moved between MI308X, MI325X and gfx950.

| choice | rule | precondition / arch |
| --- | --- | --- |
| one-shot vs two-shot vs compressed (inference, bf16) | measured: one-shot won only for the smallest messages; two-shot (reduce-scatter + all-gather) above; a compressed wire above that: a lossless bf16 code (about 12 bits per value, an escape path for outliers, bitwise-exact results) was the deployed default from mid sizes upward on gfx950 (the threshold falls as TP shrinks), and lossy codes only at the largest sizes where their acceptance passed — "compressed" is not "lossy". The one-shot crossover rises as TP shrinks — key the table on (arch, TP, size) | both arches; crossovers differ per SKU |
| push vs pull for the gather phase | measure both: on gfx950 pull (128-byte read requests) edged push (64-byte write requests) at large sizes; on MI308X push had the higher ceiling | per SKU and TP |
| CTAs per collective (inference) | a handful of CTAs already saturates the links at TP8 (one CTA pushes only a fraction of a link set); more added only interference in those A/Bs. Restricting to a CU mask: count *effective* CUs — CUs spread one per shader engine first; a mask adding a third CU on some engines added no bandwidth | gfx942 measured; check the SE layout (gfx942 SEs are non-uniform, gfx950 uniform) |
| copy engines vs CU kernels for bulk movement | engines move point-to-point at link rate with zero CUs; the measured split was engines for the bulk phases of large messages and CTAs only to reduce. Below the crossover an engine two-phase path has a high fixed device latency, and a CU one-shot / two-shot (or a fallback) won there | large messages, both arches; small messages lost on engines |
| a vendor copy-engine all-reduce | lost end to end on gfx950: slower transfer, its reduce / local-blit / wait kernels still occupy CUs, and its in-place path had a slow mid-size range — benchmark in place, the way DDP calls it | gfx950 |
| uncached workspace vs cached + writeback | measured: flags and landing slots in uncached IPC memory had the lowest latency and need no maintenance. Cached memory + `buffer_wbl2 sc0 sc1` before the flag paid, in the cases measured, only where it removed a copy (pushing straight into peers' registered outputs; cached staging for a codec pack): it won at large sizes and lost at the smallest | gfx942; a stale-line stress test found no need for an invalidate after the wait there — keep `buffer_inv` unless that test passes on your arch |
| poll primitive | plain flag store + `sc0 sc1` poll load: a hop was markedly faster than with system-scope atomics | MI308X (gfx942); on gfx950 uncached flag rows the plain form was also the fastest round trip, by a modest margin |
| fp32 vs compressed wire (fp32 gradients) | raw fp32 measured best. A bf16-exact compression wire can pay only when the gradients are bf16-exact; on real fp32 data it lost (gfx942), and on gfx950 its codec CTAs made it slower alone and heavier on the overlapped GEMMs. At TP8 the lossless heads also beat an int8 lossy wire (MI308X) | the codec's CU cost must clear the bytes it saves |
| IPC sizes with bit 31 set | `hipIpcOpenMemHandle` of an allocation whose 2 MiB-granule-aligned size is in [2 GiB, 4 GiB) hung on ROCm 7.1 (MI325X); 4 GiB and up opened. On ROCm 7.15 (a TheRock pre-release, `7.15.0a`; MI350X) such sizes opened normally. The library's guard defaults by HIP runtime version, with an override for bisection, applied to the granule-aligned size | ROCm version, not arch |
| IPC allocations | give each IPC buffer its own 2 MiB-aligned allocation: smaller allocations get packed into shared blocks and the hop latency of a latency-bound collective then varies with placement. IPC-exported memory may not be returned by `hipFree`: cache exported allocations for reuse, and never free or reuse the buffers of a collective that timed out (a late peer write may still land) | observed in a production library on gfx942 / gfx950 (source-read, not timed here) |
| messages past the IPC window | the window is capped near 2 GiB by int32 offsets: split into equal pieces (balanced beat greedy); piecing cost no throughput. A size refused registration goes to a copy path with the same arithmetic — handing it to another library silently drops bitwise rank agreement | both arches |
| copy engines per process | at most 3 engine instances per process on gfx950 (MI350X, ROCm 7.15 pre-release; 7 SDMA queues each); more failed `hsaKmtCreateQueueExt` with `HSAKMT_STATUS 6`. MI325X: 16 engines (2 PCIe + 14 xGMI), 8 queues each — the working form was one *even* xGMI engine per peer link: on odd engines `POLL_REGMEM` never released, user-mode queues were consumed late and copies were slower; a second engine per peer added nothing. MI308X: only the even engines accepted a kernel-rung doorbell (ROCm version not recorded for the MI325X / MI308X rows). Copies run slowly for a while after VM mapping changes — warm up | ROCm / SKU facts |
| summation order | sum fp32 in fixed rank order: bitwise identical on every rank and run, at every size. The vendor library switches algorithm with size and stops matching the rank-order sum above it. Fold DDP's `1/W` into the reduction only for power-of-two `W` (exact scaling) | any |
| host cost per call | at small sizes the host can be the floor: fold neighbouring 32-bit flags into one masked 64-bit stream wait (`hipStreamWaitValue64`), no per-call Python future, one reused event, a C++ comm hook rather than a Python one. These cuts together lowered host time on gfx950; the wait folding alone was neutral end to end on gfx942 | eager calls; graphs remove it |
| measuring | align ranks on the GPU before timing (one tiny all-reduce + a spin) — a CPU barrier's exit skew is counted into the max over ranks; rotate or flush inputs beyond L2 + MALL for cold numbers; interleave the candidates (card-to-card power spread was larger on gfx950) | both arches |
| hardware queues | `GPU_MAX_HW_QUEUES` (default 4) needs to cover every concurrently busy stream for them to run concurrently — compute, the DDP reducer, the collective's own, a data-prefetch stream; past it, two of them share a hardware queue and their work serializes (gfx942: a prefetch stream added to a DDP step with its own comm stream slowed it sharply; the vendor library, using one stream fewer, was unaffected) | any |

## The one primitive: an atomic carrying sem and scope

**It is the primitive the compiler checks, not the only one production uses:** a tuned 8-GPU library
publishes with plain stores into *uncached* peer memory and polls with system-coherent loads, using no
cross-device atomic at all (`## The other route: uncached peer memory, plain flag stores`). Which route
fits follows from the memory the flag and payload live in.

**`gl.atomic_{add,cas,xchg,or,and,max,min,xor}` take `sem=` and `scope=`; defaults `sem="acq_rel"`,
`scope="gpu"`.** Values: `sem` in `"relaxed" | "acquire" | "release" | "acq_rel"`; `scope` in
`"cta" | "gpu" | "sys"`. Same on 3.6-3.8.

- Default `scope="gpu"` fails silently for a peer reader: cross-device atomics carry `scope="sys"`.
- `scope` follows the reader set, not the buffer (rank-local counters can stay `gpu`; promoting all of
  them buys extra maintenance).
  Production keeps tickets and last-arriver counters inside peer-mapped buffers at `gpu` and uses
  `release`/`sys` only for counters a copy engine or the host reads.
- Disassembled `gl.atomic_add(sem="acq_rel", scope="gpu")` (Triton 3.8.0, gfx950 = gfx942): `buffer_wbl2 sc1`,
  `s_waitcnt vmcnt(0)`, `global_atomic_add … sc0`, `s_waitcnt vmcnt(0)`, `buffer_inv sc1`.
- **`atomic_scatter_add` / `atomic_scatter_max` are LDS atomics (methods on the shared-memory
  descriptor, new in 3.8.0)**, CTA scope, index-tensor form — not a cross-device primitive
  (`../gluon/smem-lds-reference.md ## Atomic RMW in LDS — the atomic_scatter_* family`).

## Publish and consume: let the atomic emit the sequence

Default when the flag or the payload lives in ordinary (cacheable) device memory — e.g. a caller's tensor
registered for peers to read, or an output a copy engine reads next. Reminder: `release/sys` emits `buffer_wbl2 sc0 sc1` + `s_waitcnt vmcnt(0)` before the flag,
`acquire/sys` emits `buffer_inv sc0 sc1` after it (`gpu` scope: `sc1` only; gfx950 = gfx942). Payload
stores + `gl.atomic_xchg(flag, seq, sem="release", scope="sys")`; consumer `gl.atomic_add(flag, 0,
sem="acquire", scope="sys")` then plain loads. This needs no inline asm; a hand-added `buffer_wbl2` next to a
release atomic is a second writeback. Asm fallback, only when the flag is not an atomic (a plain or
engine-written word): `buffer_wbl2 sc0 sc1` + `s_waitcnt vmcnt(0)` before it
(`../gluon/inline-asm/classes-sync.md ### buffer_wbl2 / buffer_inv — release and acquire`).

### Who issues the maintenance

The maintenance is attached to the intrinsic's arguments: readers fix `scope`, who completes the
writeback fixes `sem`.

| readers | writeback completed by | write |
| --- | --- | --- |
| another device | backend, from the release | `sem="release"`/`"acq_rel"`, `scope="sys"` |
| another device | **nobody** (relaxed, or default `scope="gpu"`) | the bug: fix the argument, not with an instruction |
| this device only | backend | `scope="gpu"` (default) |
| no reader in this launch | the launch boundary | keep `sem="relaxed"`; raising it buys an unneeded `buffer_wbl2 sc1` + `s_waitcnt vmcnt(0)` (`moe.md ### The launch boundary is the grid-level synchronization`) |
| another device, buffers allocated uncached | not needed (the memory type) | plain flag store after the publish fence (`## The other route: uncached peer memory, plain flag stores`) |
| — | compiler never saw an atomic (asm atomic, plain store as flag) | you: `../gluon/inline-asm-reference.md ## Class 4 — synchronization` / `## Class 4b — protocol blocks` |

Record next to every `relaxed` publish which row justifies it — the line alone cannot tell rows 2 and 4 apart.

## The other route: uncached peer memory, plain flag stores

**The route: IPC landing buffers allocated uncached** (`hipExtMallocWithFlags(..., hipDeviceMallocUncached)`,
then `hipIpcGetMemHandle`; every peer's handle opened once per process). Remote stores land in HBM and a
local poll does not hit a stale L2 line, so in the production library that uses this route this memory
needed **no `buffer_wbl2` / `buffer_inv`** (local plain stores into lines peers also write were seen stale
in production, not reproduced in a gfx950 two-GPU ping-pong: `## Rendezvous invariants` 1); what is left is ordering inside the writer and coherence bits
on the reader.

- **Writer (publish fence):** every wave drains its own stores, then the CTA barrier, then one lane
  stores the flag (a plain store on uncached memory; on cacheable memory a `.wt` store or a release-`sys`
  atomic — `## Rendezvous invariants`). No binding spells a bare per-wave drain — `gl.barrier()` does not
  wait for stores (compile-checked on 3.8.0, gfx950 and gfx942: no `s_waitcnt vmcnt` before `s_barrier`), and a release atomic is understood to drain only the wave that issues it (unverified at run time: a cross-XCD test on gfx950 saw no stale payload with barrier + release atomic and no per-wave drain) — so the fence is a small
  class-4 block (`../gluon/inline-asm/classes-sync.md ### The publish fence — drain, barrier, opaque zero`).
  API-only fallback: publish with the `release`/`sys` atomic (correct; adds a writeback).
- **Pin the flag behind the fence through its address:** add the fence block's opaque zero result
  into the flag's address. A data dependency cannot be scheduled away; `~{memory}` alone only binds the
  compiler's memory motion.
- **Reader:** poll loads need `sc0 sc1`. Compiled on 3.8.0 for gfx942 and gfx950 alike:
  `gl.load(..., cache_modifier=".cv")` and `buffer_load(..., cache=".cv")` (`gl.amd.cdna3` / `cdna4`)
  both emit `sc0 sc1` — either form is a system-coherent poll on both arches; the payload loads after a
  successful poll use `.cv` as well (GPU-checked on gfx950 / 3.8.0: plain payload reads after a `.cv` poll
  returned stale data), and the producer needs a release (or a writeback) before the flag — a `.wt` flag
  after plain payload stores let the flag overtake the payload across XCDs. Not coherent: `volatile=True` (no bits), `.cg` / `.cs` (global
  `nt`, buffer `sc0 nt`) (`linear-attention.md ### Cache modifiers on the state slab (.cg load, .wt store)`).
  A word another agent polls can be stored with `.wt`: `gl.store` and `buffer_store` both emit
  `sc0 sc1`; only the `gl.store` form adds a trailing `s_waitcnt vmcnt(0)` per store.
- **Stores and loads of one wave retire in order through one counter** (CDNA3/CDNA4 ISA: vector-memory
  reads and writes return in issue order, mixed). A wave with remote stores in
  flight that then polls waits for the remote acknowledgements inside the poll's own wait. Drain before
  the spin, or give writing and waiting to different waves / CTAs (role split) when that wait is on the
  latency path.
- **Cacheable memory in the same protocol still needs the writeback** (registered caller inputs read
  in place by peers, outputs a copy engine reads): the release-`sys` atomic as the flag, or `buffer_wbl2 sc0 sc1`
  + drain followed by a `.wt` flag store (the wbl2 + plain-flag combination was not tested, and plain flags on
  cacheable memory broke the two-GPU test; gfx950 / Triton 3.8.0).

## Transports: data as the flag, paired words, epoch flags

Three ways to tell a receiver that a block has landed; none needs a cross-device atomic.

| transport | how | precondition / cost |
| --- | --- | --- |
| **data as the flag** (sentinel) | receive slots pre-filled with a bit pattern the payload never carries (bf16 `-0.0` = `0x8000`, sent as `+0.0`; fp32 `0x80000000`); the receiver polls the payload until no sentinel is left | one pattern is lost (`-0.0` becomes `+0.0`); integers need an escape word; slots must be re-armed under a rotation proof (`## Rendezvous invariants` 3); no flag word, no fence |
| **paired generation words** | each payload word written together with its generation as one aligned 8-byte store; accept when the generation matches | the link must deliver the aligned pair indivisibly (verify on the target); doubles the bytes; keeps every payload bit |
| **epoch flag per block** | payload stores, publish fence, then a flag = generation per block | for payloads with no free pattern (quantized codes, scales); the fence and flag amortize over the block |

Production polls every source of a step in one loop iteration — all loads issued, one wait, the compares
folded (e.g. a packed min over half-words against the sentinel) — so the sources' latencies overlap instead
of adding up one source at a time.

## Spinning, and the hoist that will eat your wait

Spin on an `acquire`/`sys` atomic read or on a `.cv` load; a plain-load spin is hoisted. Compiled on
3.8.0 (gfx942 = gfx950): a plain or `volatile=True` load of a uniform flag address becomes one scalar
`s_load_dword` above the loop **even inside `while tl.condition(cond, disable_licm=True)`** — the spin
never re-reads (GPU-checked on gfx950: plain and `volatile` polls timed out in every producer form and XCD
placement); `volatile=True` adds no cache bits either. A `.cv` load stays in the loop as
`global_load … sc0 sc1` (with or without `disable_licm`); the atomic poll compiles to
`global_load … sc0 sc1` + `buffer_inv sc0 sc1` per iteration. Keep `disable_licm=True` as belt and braces
— `tl.condition` is not in `gl.*` (3.6-3.8); import `triton.language as tl` inside `@gluon.jit` — and
read the loop body in the ISA.

**Production form: the whole poll loop inside one protocol asm block** (no LICM can touch it; all
sources polled per iteration): `../gluon/inline-asm/classes-sync.md ## Class 4b — protocol blocks`.
**Bound every spin by wall-clock time**, not only by iterations: read `s_memrealtime` every few thousand
failed polls, on expiry leave with a reserved counter value, report it, and optionally `s_trap` (HIP then
reports it as "Memory access fault … (nil)" — check peers, call order and the bound before chasing an
access bug); an
optional host-written abort word (read with an uncached scalar load) lets the host end a hung
collective. `gl.*` has no clock or trap builtin, so an iteration-count bound is the API-only fallback.

## Co-residency is a launch option, not an assumption

**Triton has a cooperative launch: `kernel[grid](..., launch_cooperative_grid=True)` routes through
`hipModuleLaunchCooperativeKernel` (all blocks co-resident).** 3.7.0+ asserts
`device_properties['cooperativeLaunch']`; 3.6.0 fails at the HIP call; an oversized grid fails with
`hipErrorCooperativeLaunchTooLarge`. Without it a block spinning on an unscheduled sibling deadlocks.

Structures that avoid needing it (the surveyed production stacks use these, not the option; in the pinned
aiter snapshot the checked residency limit and the per-block replicated rendezvous were located, 1 and 3 were
not — they remain options):
1. **Cross-block wait in its own `grid=(1,)`, `num_warps=1` kernel** (one block is trivially resident;
   not the case in `prologue.md ## A 2D grid with a small axis 0: grid shape, not num_warps=1`).
2. **Residency as a checked number** (host-side raise if blocks > CUs, or a documented grid clamp).
   Derive it from the *compiled* kernel's occupancy (registers / LDS → CTAs per CU) with margin, for every
   build form you ship (an instrumented build can cross a register step and halve the limit). A production
   library budgets half of every CU — at most `min(2, ctas_per_cu // 2)` waiting CTAs per CU — so two
   waiting kernels running concurrently (two streams, two communicators) cannot starve each other; it
   counts CUs per device (SKUs of one arch differ), not per arch.
3. **Last-arriver election, losers exit** (they must exit, not spin): `old = gl.atomic_add(cnt, 1,
   sem="acq_rel", scope="gpu")`; `old == N - 1` elects.
4. **Replicate the rendezvous per block** (signals indexed `[block][rank]`; block *i* only waits on
   block *i* of each peer) — grid unconstrained.
5. **Let the copy engine do the waiting.** In a device-initiated copy-engine collective one wave writes the
   engine packets — including engine-side poll-on-memory waits for peer readiness and for reducer completion
   — and rings the doorbell; no CTA then waits for another CTA, so the launch needs no co-residency
   (production library, gfx942 / gfx950).

Otherwise split the kernel and let the launch boundary synchronize.

## Rendezvous invariants

Unchecked by the compiler; they fail only under unlucky arrival order. Reminders:
1. Flags in their own allocation, never in a payload line (the acquire's invalidate discards a co-located
   payload). **Whether a flag store becomes visible depends on the memory type; line layout alone does not
   fix staleness** (two-GPU ping-pong, gfx950 / Triton 3.8.0, `.cv` polls):
   - **Cacheable memory** (ordinary `hipMalloc`): every flag store on both sides — local and remote — is
     `.wt` or a release-`sys` store. A plain store, local or remote, broke the protocol even with each flag
     on its own line.
   - **A `.wt` flag is not a release:** across XCDs it can overtake plain payload stores, so publishing a
     cacheable payload needs a release (release atomic, or `buffer_wbl2` + drain before the flag).
   - **Uncached memory** (`hipDeviceMallocUncached`): plain, `.wt` and release-`sys` stores all worked in
     every layout tried, including a local plain store into a line peers also write. Production observed
     that last case returning stale peer bytes to a later `sc0 sc1` poll (all-to-all count rows, arch
     unrecorded), so one writer per line (128 B on both arches) stays a precaution there, not a fix.
   Several writers' flags sharing one line (one row per block, a slot per rank, so one load polls every
   peer) is fine under these store rules; line-per-flag costs nothing extra when lanes poll sources in
   parallel.
2. Every wave drains its payload stores (`s_waitcnt vmcnt(0)`) before the CTA barrier that precedes the
   one-wave publish — `gl.barrier()` alone does not wait for stores, and a release atomic is understood
   to drain only its own wave (unverified at run time, see `## The other route`); a barrier again after
   the acquire.
3. Monotone sequence numbers, not toggled bits. A signed difference `>= 0` is wrap-safe only between
   values that were both written: a flag never written still holds 0, and once the generation passes
   2^31 it satisfies `flag - gen >= 0` at once (seen in production: a flag index first used late in an
   instance's life returned before its data landed). Where the protocol guarantees a flag holds this
   call's value or an older one (no peer starts call n+1 before this rank ends call n), use a never-zero
   generation (`cnt mod (2^32 - 1) + 1`) and wait for equality. A reset is itself a publish: legal only
   with a rotation proof — e.g. sentinel slots rotated over K calls, the receiver re-arming the slot of
   call n−1 during call n, which no peer writes again before call n−1+K.
4. One owner initializes each buffer before the first launch; record the world size.
5. Every rank takes the same branch: derive fast-path refusals, size guards and registration decisions only
   from data every rank holds (or agree on them over the CPU group first) — one rank diverging leaves the
   others spinning.

**Under graph capture:** protocol state lives in device memory (flag starts at 1, never reset); the
peer-pointer table is a stable allocation filled after capture (IPC registration is illegal inside capture).

**Claiming work:** `old = gl.atomic_or(bitset_ptr + w, 1 << b, sem="acq_rel", scope="sys")`;
`mine = (old & (1 << b)) == 0` (RMW returns the old value). On a failed claim advance, never retry
the same slot.

## Peer pointers arrive as integers

**Peer base addresses come from the host as an integer tensor and are cast with `gl.pointer_type`**
(`../gluon/imports-and-launching.md ## Pointers as data (gl.pointer_type)`). A peer address is opaque
to masks/bounds/`gl.max_contiguous`; a stale entry (reallocated peer, wrong world size) faults as an
illegal access with no hint which entry. Build the table once on the host; read-only in the kernel.

- **A pointer loaded from the table carries no alignment fact** for the compiler: `gl.multiple_of(p, 16)`
  states it (only if the host registers 16-B aligned addresses). Without it wide peer loads can split into
  dword loads; a compile-only check on 3.8.0 / gfx950 with a uniform table pointer still got
  `global_load_dwordx4` without the hint, so read the ISA of the actual kernel.
- Alternative: pass the bases (or the rank) as `constexpr` integers (one binary per rank). Then each
  rank compiles its own floating-point program and the backend may contract a different mul+add into
  an fma per rank — observed as a one-ulp rank disagreement in a fused norm's `rstd` at TP8. Write
  order-sensitive arithmetic as explicit fma chains in source order and gate it on the ISA (same fp
  instruction sequence in every rank's binary) when ranks must agree bit for bit.

## How to verify any of this

1. **Zero `buffer_wbl2` in the source is correct when the publish is a `gl.atomic_*` with
   `sem`/`scope`** — the backend emits it — **or when flag and payload live in uncached memory**. Grepping for files without maintenance flags every correct
   kernel; a hand-written writeback is only a finding if the signal is an asm block or not an atomic.
2. **Disassembly must show `buffer_wbl2 sc0 sc1` + `s_waitcnt` at a `release/sys` publish and
   `buffer_inv sc0 sc1` at the consume**; absence is a finding on either path. Uncached route: every
   path from a publish fence reaches the flag store as its first store, and the polls carry `sc0 sc1`.
3. **Compile-only ISA gates, no GPU:** (a) after every publish fence the first store on every path is
   the flag (its address depends on the fence's zero) — no payload store slips behind it; (b) no vector
   store is outstanding when a spin starts; (c) every asm block that writes SCC declares `~{scc}`; (d) every
   API-form poll re-reads inside its loop as a vector load with `sc0 sc1` (no `s_load_dword` of the flag
   above the loop). Keep a control kernel with a deliberately misplaced store that must fail each gate.
4. Race-test with varied arrival order
   (`../method/benchmark-hygiene.md ## Determinism race-test (async / barrier / pipeline / layout changes)`);
   a green race test proves no race fired, not sufficiency. Test at a world size you did not develop
   at (TP=8 exposes what TP=2 hides).
