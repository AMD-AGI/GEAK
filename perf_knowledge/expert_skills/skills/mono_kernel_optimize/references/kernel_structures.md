# Kernel structures: classify the kernel before you plan

A lever only makes sense for a given structure. Collect the five facts below, write the class into your
notes, and only then open the lever references.

Status of this file: the classes are an index of structures that existing mono kernels use (*read from
source and design notes*, not re-measured here). The "what follows" columns are derived from the design
unless they cite [measured_rounds.md](measured_rounds.md). The classes are structural: never pick one
from a model name.

## Five facts to collect (read; do not run anything yet)

| fact | where to read it |
|---|---|
| launches per layer per rank | the wrapper / adapter the unit test calls: count device-kernel launches in one layer call. Cross-check later in a kernel trace: kernels per decode step ÷ layers |
| coverage | the kernel's header comment and stage list: does it start at the input norm (full) or at the attention output (partial)? Are both all-reduces inside? |
| grid and threads | grid size against the CU count (304 on MI300X); threads per workgroup (256 or 512) from the launch bounds / known block size |
| schedule inside the launch | how a workgroup picks its work: every stage in order, a per-stage task table, or two wave groups |
| rows | whether stages take a row count (several decode rows per step), and which widths the unit test's cases use |

## Layer-level class

| class | how to recognise it | what follows |
|---|---|---|
| **full, one launch** | one launch per layer per rank covers input norm → attention → FFN / MoE; both all-reduces inside it; hand-offs are per-tile tags under one epoch | Every lever family applies. The launch count is already minimal: all remaining gain is inside the launch (overlap, hand-offs, streams). |
| **dual launch (front + tail)** | two (at most three) launches per layer with fixed roles: a front launch writes the KV cache / key table, the tail launch runs the attention core, FFN / MoE and both all-reduces. The boundary is a device-level release (the launch boundary), never an in-kernel tile wait | Legitimate for a deployed kernel: it exists when one launch cannot hold attention + MoE in registers and LDS, or when the KV write needs a device-level barrier before attention. Do not add a third launch. Merging the two is a multi-round structural project (register / LDS budget; the KV write must be visible before the last block is read), not a first-round lever. Measure each launch separately (timestamps compare only within one launch) and apply levers inside each. A timed-out wait in either launch fails the whole step. |
| **partial, from the attention output** | the framework still runs attention (or another token mixer); the kernel starts at o_proj / the post-attention norm and runs FFN / MoE and the all-reduce(s). Its first stage consumes an external tensor, not the input norm | Attention levers do not apply. Stage balance (task-count ratios) is computed from o_proj on. The framework boundary is a tensor port: no in-kernel hand-off can precede it, but weights can still be issued at kernel start. |
| **multi-launch layer** | four or more launches per layer | Launch boundaries dominate. The structural lever is fewer boundaries, taken in steps that each pass the oracle; hand-off levers apply only inside a launch. A structure reference, not a template. |
| **multi-layer per launch** | one launch runs two or more decoder layers | Layer-to-layer hand-offs are inside the grid; "advance the epoch once per step" becomes "once per launch". A timeline must tag its stamps with the layer. Outside the single-layer lever set: plan from the timeline. |

Also seen next to these: a **tail launch** outside the layers (final norm + LM head, sometimes a logits
gather). It is not part of the layer kernel; leave it alone unless the task names it.

## Schedule inside a launch

| schedule | how to recognise it | consequences |
|---|---|---|
| **every workgroup runs every stage** | one resident workgroup per CU, usually 256 threads (one wave per SIMD: up to 512 registers per wave — 256 architected VGPRs plus the accumulator file, which is reached only through moves); each takes a fixed share of each stage in order and waits only on the tiles it reads; no device-wide barrier | The default for decode layers; each stage may use all 64 KiB of LDS. Overlap comes from issuing the next stage's loads early. Weak point: a stage whose tiles finish unevenly (attention by context length) delays every workgroup behind it. |
| **per-stage task tables** | each stage is a table of tasks; task i runs on workgroup `(base + i) mod grid`, then every grid-th; bases differ per stage; usually 512 threads (two waves per SIMD: 256 registers per wave) | Fits layers whose stages differ in task count by about an order of magnitude (a workgroup per token for norms, hundreds of GEMV tiles, a few attention heads). Levers: bases, readers per token, placement by XCD. |
| **two wave groups per workgroup** | 512 threads; code branches on the wave index into a long-stream side (QKV, attention, gate/up) and a short-dependent side (norms, o_proj, router, down, all-reduce) | Lets one long stage overlap the short ones. Halves the registers per wave (256 cap; spills start there) and splits the LDS between the sides, which talk only through tagged hand-offs. Not the same thing as a dual launch. |

Several decode rows in one step come in two designs: **rows as MFMA columns** (shared-weight GEMVs load
each weight tile once for all rows) and **workgroup groups per row** (each group runs the 1-row flow and
re-reads the dense weights). Details: [structure_levers.md](structure_levers.md).

## Task shape

Decide by the source first, then confirm with the unit test's baseline leg. If in doubt it is (a): keep
the launch count the kernel arrived with.

| | (a) deployed mono kernel | (b) freshly fused seed |
|---|---|---|
| source | a framework package layout; HIP C++ or FlyDSL with its own sync helpers | an adapter plus a kernel from an authoring flow, with a vendored synchronization runtime |
| baseline leg | usually the same mono kernel as extracted (first speedup about 1.0); a deployed kernel timed against the native layer is still (a) | the framework's stock layer, many kernels per layer (first speedup may be below 1.0) |
| launches | as received (one, or front + tail) — never more | exactly one per layer per rank |
| levers | many are already applied: inventory first | few are applied: the standard order of attack holds |
| rules | universal rules | universal rules plus the seed-only rules |

Both rule sets: [acceptance.md](acceptance.md).

## Concept map: the same idea in three vocabularies

A fused seed speaks its generation flow's runtime words; a deployed FlyDSL kernel brings helpers like
the public ones in the third column; a deployed HIP C++ kernel has the same ideas as plain device
functions. Public symbols were read (names and docstrings only) at the commits in the skill's Sources:
they illustrate the vocabulary and are not the API of your task's kernel — read its own helpers.

| concept | generation-flow runtime words | public FlyDSL runtime: sgl-amd-mono `sgl_amd_mono/common/`, FlyDSL `kernels/monokernel/` | in a HIP C++ kernel |
|---|---|---|---|
| stage plan | stage, tile, task, CU plan | `plan.py`: `first_task(bid, base)`, `Shard`, `pair_layout`; FlyDSL `layout.py`: `stage_tasks` | one device function per stage, called in order by the top-level kernel; often a second multi-row family of them |
| resident grid | one persistent workgroup per CU, known block size | `plan.py`: `BLOCKS`, `THREADS`, `WAVES` | launch bounds sized so one workgroup fills a CU; grid ≤ CU count |
| launch epoch | sequence number read at launch begin, published at launch end | `epoch.py`: `epoch_begin` / `epoch_end` (every workgroup marks the epoch; workgroup 0 moves it on once all have) | an epoch word in device memory, read at entry and advanced when the last workgroup is done; the kernel receives its layer index and the layer count |
| tagged hand-off | mailbox put / get; or data, drain, then flag := epoch | `sync.py`: `Mailbox` ("each value is stored next to the launch tag"), `publish` (drain the plain stores, workgroup barrier, then store the pair) | a "tagged pair": a 32-bit payload and the launch tag in one 64-bit store; a poll helper spins until the tag matches |
| sentinel words | not used | `sync.py`: `put_sent` / `poll_sent`: plain words where "unwritten" is a reserved bit pattern the reader re-arms — no tag, half the bytes of a pair, nothing to zero; a word must not be pushed again before its reader re-armed it, so the caller alternates two copies | read the kernel |
| scopes | agent / device scope inside a GPU, system scope across GPUs | `sync.py`: `Space.SCRATCH` (device scope), `Space.PEER` (system scope) | separate small helpers for device-scope and system-scope 64-bit loads and stores |
| batched poll | many tags per lane, reissue the stale ones | `sync.py`: `POLL_MAX` specs per batch (it bounds the live registers) | the poll helper takes a range of tags |
| device-wide barrier | grid barrier | none | usually absent in a tuned kernel; if present, a counter every workgroup waits on |
| cross-rank exchange | cross-rank barrier, all-reduce sum, double-buffered partial slot | `sync.py`: `sent_sum` (the ranks' words "summed in fp32 in rank order"); FlyDSL `symmetric_allreduce.py`: `build_symmetric_bf16_allreduce`, `build_symmetric_bf16_allreduce_rmsnorm` | an all-reduce helper per tile: push pairs into every rank's uncached peer buffer, poll, sum in rank order in fp32 |
| peer memory | peer group, control region, IPC-shared uncached buffers | `peer.py`: `PeerBuffer`; FlyDSL `runtime.py`: `SymmetricPeerBuffer`, `ipc.py`: `allocate_uncached`, `get_allocation_base` | per-rank pointers in the argument struct: a workspace of pairs, peer buffers, epoch / done / error words |
| bounded wait | wait-until with a deadline, sticky error word, a site id per wait | `guard.py`: `Guard`, `deadline_ticks`, `stop_launch`, sticky error words, a check every `CHECK_POLLS` polls | the poll helper compares the realtime counter with a tick budget, then sets the error word and aborts |
| timeline | a profiling build with its own kernel name and one extra buffer argument | `stamps.py`: `stamp(c, point)`, `timeline_row`, `TL_POINTS` — a build without the scratch region records nothing | an optional trace pointer in the argument struct (null in production) and a stamp helper or macro, thread 0 only |
| tag wrap | size the tag, or clear at a known point | read the kernel | a clear routine every rank runs after a fixed number of steps |
| rebuild key | a digest of the sources in the kernel name | `plan.py`: `kernel_sources`, `key_tuple` | the build system's own dependency tracking |

## Which sources you may read

| source | use it for | not for |
|---|---|---|
| the framework's native implementation of the layer | what the baseline computes: reduction order, rounding points, routing rules | — |
| single-operator kernels of an operator library | how one operator is tiled and streamed | — |
| model-independent layer primitives (runtime, layout, all-reduce) | vocabulary and synchronization patterns | — |
| a different model's finished mono kernel | structure and optimization ideas only | numerics, constants or shapes of your model |
| the target model's own finished mono kernel from any source other than the task directory (a framework package installed in the environment, the other framework's package, a newer upstream version, a different branch) | nothing: do not open it | — |

Why the last row: copying it is not an optimization result (the lane measures what you changed), and
its numerics and argument layout belong to a different integration. In shape (a) the task's own kernel
source is of course what you edit; the rule is about other copies of it.
