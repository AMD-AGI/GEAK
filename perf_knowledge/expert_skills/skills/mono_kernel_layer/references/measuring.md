# Measuring where a persistent layer kernel spends its time

Always measure before choosing a lever: without per-stage times and per-stage floors, every other lever
is a guess. The oracle's own numbers (what a per-case time means, the weighted speedup) are defined in
[acceptance_gates.md](acceptance_gates.md#what-the-numbers-mean).

## Per-stage timeline

**When.** Always first. A rocprofv3 trace shows one kernel per layer, so it cannot tell which stage is
slow.

**Recipe.**
1. Build a **profiling variant** of the launcher (separate `func.__name__`, e.g. suffix `_prof`, so the
   compile cache never mixes it with the real one) that takes one extra `Int64` argument: a device
   buffer of `n_blocks × n_events` u64.
2. In every workgroup, thread 0 stores `d.realtime()` (`s_memrealtime`, 100 MHz = 10 ns ticks, the same
   clock on every CU) at fixed events: kernel entry, start and end of each stage it works on, before
   and after each wait (mailbox, grid barrier, cross-rank barrier). Store with a plain store; the
   buffer is read after the kernel finishes.
3. Run the variant in a small driver of your own, under a whole-group lease (see
   [acceptance_gates.md](acceptance_gates.md#running-the-oracle)), one process per rank, through the
   adapter on the captured inputs; replay a captured graph ~20 times and keep the median per event.
4. Per stage: span = max(end) − min(start) over its workgroups; wait time = time spent inside waits. Per
   launch: entry skew (first vs last workgroup entry) and the tail after the last stage.
5. Next to each stage's span, write its **floor** (next section).

**Reading it.**
- A stage far above its floor → that stage's kernel
  ([weight streaming](stage_kernels.md#stream-the-weights-at-hbm-speed),
  [split-KV attention](stage_kernels.md#split-kv-decode-attention)).
- Stages near their floor but the layer still slow → structure: imbalance
  ([CU rebalance](layer_structure.md#rebalance-the-cu-plan)), waits and barriers
  ([per-tile hand-offs](handoffs_and_sync.md#per-tile-hand-offs-instead-of-stage-wide-barriers)), the
  all-reduce ([all-reduce fusion](handoffs_and_sync.md#fuse-the-tp-all-reduce-into-its-neighbours)).
- Large entry skew → too many workgroups or heavy per-workgroup setup before the first stage.

**Pitfalls.**
- Timestamps cost a few ns each; keep them out of the real launcher (the variant is a separate build).
- Compare ticks only within one launch; `realtime` is not synchronized across GPUs.
- Do not add the buffer argument to the real kernel "temporarily": gates 5 and 7
  ([acceptance_gates.md](acceptance_gates.md#hard-gates)) apply to what is installed.

**Check.** The timeline is a measurement, not a change: nothing to gate. Keep the driver and the variant
under `kernel_src/` only if the gates still pass with them (the static rules apply to every file there);
otherwise keep them in your scratch directory.

## Floors

- **Byte floor of a stage** = the bytes it must read from HBM ÷ ~4.5 TB/s achievable on MI300X (5.325
  TB/s peak, [`hardware/cdna3_mi300/arch.md`](../../../../hardware/cdna3_mi300/arch.md)). The bytes are the
  weights for its share of this rank (scales included) plus, for attention, the KV at the case's context.
- The harness flushes L2 and MALL before every sample, so assume HBM, not the 256 MiB Infinity Cache.
- The floor of the layer's GEMVs is the sum of this rank's weight bytes (shapes and dtypes from the stage
  list, scales included) ÷ ~4.5 TB/s. Compute it before deciding that a GEMV stage is slow.
- At decode width 1–8 every weight byte is read once per step and used for a handful of FLOPs, so the
  layer is HBM-bound.

## Sources

- Hardware figures: [`hardware/cdna3_mi300/arch.md`](../../../../hardware/cdna3_mi300/arch.md).
