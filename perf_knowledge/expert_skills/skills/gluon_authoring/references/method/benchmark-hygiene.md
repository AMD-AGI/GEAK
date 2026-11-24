# Benchmark hygiene — stack-specific traps (gfx950 / ROCm 7 / Triton 3.8.0)

Advisory. GEAK's frozen task commands (`COMMANDMENT.md`, verify, `MIN_IMPROVE` commit gate) own
acceptance; this page lists only the measurement traps on this stack a careful agent still hits.
Screening A/B tool: `kernel_workflow/scripts/kernel_tools/ab_bench.py` (`--task DIR`, `--cache cold|hot` — no `auto`, default
`cold`; `--control`, `--permute`, `--preload SEC`, `--budget-ms`). Device selection:
`kernel_workflow/scripts/gpu_lock.sh <gpu> <cmd>`.

## Profiler output hygiene (do not pollute the repo)

- **`rocprofv3` run with CWD inside a repo leaves a root-owned `<host>/*.db` dir + `gpucore.*`** a
  non-root user cannot delete. Run from a scratch dir (`/tmp/prof_<run>`), point `-d` outside the tree.
- **Pass `-f csv` / `--output-format csv`**; the default is SQLite `.db`, which the parsers do not read
  (`profile_kernel.sh` already sets `RPV3_TRACE_ARGS=--kernel-trace --stats --output-format csv`).
- **A JSON config passed as argv through `docker exec bash -lc '…'` is re-split → rc=4 /
  `JSONDecodeError`.** Pass it as an env var holding a file path.
- `dump_ir.sh` runs the compile command verbatim: pass only arguments the benchmark entrypoint accepts
  (a correctness-only flag such as `--check` that it rejects makes the compile fail and leaves an empty
  static op mix).

## Device identity: HIP_VISIBLE_DEVICES vs ROCR_VISIBLE_DEVICES

**Never inline `HIP_VISIBLE_DEVICES` into a timing/profiler command; run under `gpu_lock.sh`.**
- A visible-devices variable set inside an already-constrained context **re-indexes against the
  visible set** (override, not sub-selection): ordinal 0 silently lands on another physical GPU.
- `HIP_VISIBLE_DEVICES` inside an outer `ROCR_VISIBLE_DEVICES` is re-indexed against the ROCR set, so
  the same physical index in both leaves the child with no device (it failed at once on gfx950; earlier
  recorded as a rocprofv3 "TCC hang"; `profile.md ### Wrapper conventions`). For a profiler child set one
  of them with the right index, not both with the same index (`rocprofv3_safe.sh` resolves this).
- **Three index namespaces disagree**: SMI `cardN` (PCI-bus order), `/sys/class/drm/cardN` names, and
  the runtime ordinal (rank of drm entries; a display adapter may occupy the lowest drm index and be
  absent from the GPU list). Measured: disagreed on all 8 GPUs of one node. Anchor per-device claims to
  a UUID / PCI bus address (`scripts/gpu_identity.py` for ISA identity).
- A memory-free guard passes a device saturated by foreign compute; require compute-idle **and**
  memory-free, sampled over a window (a 3 s job was invisible to a 3–4 s census).
- **An SMI `GPU(s)` column beside a PID list can be a count, not an index** (four processes each
  showing `1` is not "all on device 1"); cross-check against per-device utilisation before taking a
  device. **A PID→device map built from two separate tool calls is a race** (PIDs churn between calls
  and the mismatch looks like a namespace bug) — collect both sides in one pass.

## Clock / DVFS on partitioned parts

- **`rocm-smi --setperflevel high` can return "Not supported" on partitioned cards** — you cannot pin
  the clock; use interleaved A/B (median of per-sample ratios) and clock-insensitive counters
  (`MfmaUtil`, `VALUBusy`, `MemUnitStalled`; `profile.md ## Derived metrics`).
- **Slot effect on an idle card:** the first timed cell can run on a decaying clock excursion. Observed in the
  upstream pack's measurement (a large first-slot excess that `ab_bench.py --preload 3`, 3 s of real compute
  before the first cell, removed); not reproduced at that size on MI350X (gfx950) with this harness, where any
  first-cell excess was small and present with or without preload. Keep `--preload` as cheap insurance and read
  the first cell against later ones rather than assuming a size. `--warmup` warms the variant, not the card.

## Artifact and cache hygiene

- Separate `TRITON_CACHE_DIR` per variant; same module/kernel name with different bodies = assume cache
  pollution until a fresh-process control says otherwise. Prewarm every timed config key on both arms.
- **One variant per process when a variant monkey-patches the toolchain** (a re-injected pipeliner,
  a backend hook, a `triton.knobs` toggle): the second import wins and both arms silently measure the
  same pipeline — `fingerprint()` and `--permute` cannot see it.
- **Same binary is proved by a normalised-asm diff, not by matching counters.** A forked probe that
  missed four `constexpr` defaults matched the shipped kernel on nine ISA counters (MFMA, LDS, VALU,
  loads, registers, spills) while being a different program; counts are order-insensitive. Compile both
  at the production config, strip symbol names / labels / comments, diff line by line — or call the
  production entry point instead of rebuilding its launch arguments.
- **Config sweeps: pre-screen LDS, catch `OutOfResources`.** An LDS-overflow config compiles and raises
  a catchable `OutOfResources` at first launch on Triton 3.8.0
  (`../hardware/planning-constants.md ## Make an occupancy table falsifiable before you quote it`).
  Pre-screen staged bytes against the arch cap (gfx950 160 KiB/CU; gfx942 64 KiB/CU; `hw_budget.py
  --brief`); keep one config per subprocess anyway when variants patch the toolchain (above).

## Measurement basis + benchmark-artifact pitfalls

- **Cache eviction: read-evict is the safer default; write-evict can perturb the timed kernel.** A 512 MiB `zero_()` write-evict leaves dirty
  lines whose writeback contends with the timed kernel: on a decode-weighted fused-MoE (MI355X) a
  write-evict flush read clearly slower than read-evict, which sat close to no evict; re-confirmed on MI350X with
  a streaming elementwise kernel.
  `harness_lib.flush_cache` read-evicts; check yours by timing one kernel under each flush mode.
- **Cold vs hot can reverse the winner, not just scale it.** On a per-token quant kernel hot timing
  preferred the tuned config, cold (deployment) preferred the shipped one. Screen with
  `ab_bench.py --cache cold`, one launch per sample; hot timing is a diagnostic (what the cache hides),
  not a screen.
- **Fresh-process variance (gfx950):** small on a prefill bucket, **large on a small-M decode bucket**
  — larger than a real few-percent win, so one unpaired decode pair decides nothing; use interleaved
  pairs and read the spread of repeated identical arms first.
- **Per-sample sync, not batched queues:** batching samples without a sync raised per-window overhead
  in most configs and let event and wall time disagree on kernel time under flush; sync per sample and
  check that event and wall agree.
- **Event timing silently becomes host timing on a host-bound segment** (stream idle while the host
  enqueues). Tell: time flat across a shape sweep. `harness_lib` reports `primed`/`host_ms`; remedy is
  `inner > 1` or graph replay. Sanity: `issue_only <= wall`, `device_events <= wall`, a graph arm
  normally not slower than eager (a slower one is a harness signal).
- **Launch-bound test:** when issue-only time exceeds roughly 0.8 of the synced wall, the segment is
  launch-bound and fewer dispatches is the matching lever family. **Dispatch-merge savings are sublinear** (the survivor
  carries both kernels' arguments): a merge is priced by timing both configurations; `per_launch × launches_removed`
  over-predicted it (re-confirmed on MI350X with tiny-kernel merges).
- **A fixed-graph A/B gives each arm one allocator placement**, which shows up as a stable, one-sided
  per-shape bias that reads like a per-shape pattern. Run an identical-arm null (same kernel in both
  arms) through the same harness first; a per-shape table built on an un-nulled harness describes the
  harness (`dispatch-table.md`).
- **Interleaving cancels only common-mode drift.** Host contention is not common-mode when the arms'
  host/device ratios differ, so an interleaved A/B still moves with host load. Suggested: stamp the host
  load (load average) on every recorded ratio, and compare ratios only within one load band.
- `tensor.item()` forces a GPU→CPU sync (stream drain plus host round trip) — on a sub-ms bucket that can
  erase a kernel win; time it on the target host.
- Launch-fusion wins measured eager can be inert under a captured graph; measure at the serving
  boundary.

### Instrument floor: an empty kernel through the same protocol

**Time an empty kernel (one program, immediate return) through the same timer/protocol; that is the
floor `F`.** `F` depends on **launches per timed sample**, not on the flush (GPU-measured, MI350X): one eager launch
per sample gives the largest floor, warm or cold alike; one graph replay a lower one; a kernel inside a
multi-kernel graph the lowest. The upstream pack's cold-vs-warm floor pair is the one-launch vs
many-kernels-per-window split, not a cache effect. Measure `F` with the protocol you report; bookkeeping
kernels that read near `F` cost far less than they read. Within ~3F, quote `(t - F)` ratios or nothing (subtracting `F` changed a speedup's size and turned an
"at ceiling" bandwidth census into one well below the ceiling). A one-kernel graph replay lowers this additive term but does not remove it.

## Check the instrument before you read the scale off it

A fitted reference (regression, fitted roofline, intercept-as-floor) is not a sound basis for
extrapolation while its residuals are structured and of the same order as the effect; the residuals
show which. A
decomposition whose parts sum past the whole has either a per-sample fixed cost (gap grows with the
number of splits) or lost overlap from isolated capture (gap does not move when re-cut).

## Cross-version ratios: attribute the move before claiming it

Suggested: print both arms' **absolute** times per Triton version; the baseline moves too. Observed
(recorded on 3.7.1 → 3.8.0, not re-run): one attention plain regressed sharply while its Gluon anchor moved
little, so the ratio jumped with no anchor gain. Carry a `plain@num_stages=1` third arm to separate pipeliner regression from a mistuned config.
On the Gluon arm `num_stages` is consumed by no 3.8.0 pass — it is not a tuning knob there. A baseline
that fails to build on one version (LDS limit) is recorded as such, not substituted.

## A compile-only kill step is a falsifier, not a price

Normalised instruction stream unchanged ⇒ stop (no timer can separate the arms). Same mix in a different
order, or a changed mix ⇒ nothing follows about time; run the timed leg. **Compiled ISA is per-warp code: cross-warp replication has no instruction to
count**, so an ISA diff for it returns an uninformative zero. Read replication from the layout object:
`rep = 2 ** (number of zero warp bases)` in the linear layout (one enumeration: a rule that held on
two shipped points failed in a large minority of cases).

## Before timing, list every quantity the edit moves

Widening a per-lane access removes narrow accesses **and** covers more of the reduction axis per step
(bytes read go up): a null is consistent with two cancelling effects. Bound each co-moving quantity
or re-cut the arm.

### A knob that is off still has to prove the shipped stream did not move

**A default-off `constexpr` branch can change the shipped config's instruction stream.** Usually it is
pruned to identical ISA (compile-checked on a plain GEMM loop, 3.8.0 / gfx950), but one record showed a
spelling that hoisted two operand loads added **6 `s_waitcnt`** to the shipped config while VGPR/AGPR/
SGPR/spill/LDS and every arithmetic, `ds_*`, `buffer_*`, MFMA count stayed identical and output stayed
bit-exact (cause not confirmed). Suggested guard: hash the filtered `.s` (strip directives, local labels, comments) against the same
source with the branch **deleted** (not predicate=0); keep `s_waitcnt` in per-opcode diffs.

## A pre-registered criterion fixes the threshold, not the statistic's power

Before using a statistic as a criterion, write what it reads under each alternative you want to rule
out; if both give the same sign and magnitude, it cannot decide.

## Served-range sweep

Sweep the full served range of the varying dimension (decode batch, prefill chunk) at the production
boundary, including the transition ridge where the winning tile flips (found by sweep) and points
just below/at/above each dispatch cutoff. Feeds `dispatch-table.md`.

## Determinism race-test (async / barrier / pipeline / layout changes)

**Same seed and inputs, N ≥ ~40 launches, `max |out_i - out_0| == 0`, with the output buffer
re-allocated (or sentinel-poisoned) each launch** — into a reused buffer the test passes even when the
kernel writes nothing (one harness reported 4/4 CLEAN on such a kernel). Required for barrier elision,
`wait_group` depth, relaxed loads, ring depth/indexing, recovered layouts. Green means no race
*fired*; it is zero evidence that an elided barrier is sufficient — read the ISA for that. Run beside
the oracle, not instead (a mis-recovered layout is deterministic and wrong).
