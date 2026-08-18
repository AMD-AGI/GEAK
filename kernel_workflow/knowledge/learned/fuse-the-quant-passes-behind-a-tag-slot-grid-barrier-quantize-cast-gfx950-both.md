---
name: fuse-the-quant-passes-behind-a-tag-slot-grid-barrier-quantize-cast-gfx950-both
description: Fuse a 3-dispatch dynamic per-tensor quant into one kernel behind a 2-round-trip tag-slot grid barrier: 1.73x weighted, every case up.
keywords: ['quantize-cast', 'fp8', 'dispatch-collapse', 'kernel-fusion', 'cross-workgroup', 'arrival-counter', 'coherence', 'raw-hip', 'latency-bound', 'grid-occupancy', 'profiler-error', 'paired-ab-rig', 'gfx950']
kernels: ['dynamic_per_tensor_quant', 'fused_dynamic_per_tensor_quant_kernel']
platforms: ['gfx950']
kernel_class: quantize_cast
regime: both
key: dynamic per-tensor activation quant to fp8 that ships as reset + absmax + quantize passes, HIP/C++ source on gfx950/CDNA4, timed under graph replay
layer: learned
levers: ['algo.fusion', 'host.dispatch-count', 'algo.grid-barrier']
cost: L3
lifecycle: active
type: lever
confidence: ★★
effect: 1.73x weighted director-verified vs the frozen baseline, every case up (roughly 1.49x on the largest / 1.72x mid / 1.96x smallest, i.e. the win grows as the fixed cost dominates); paired, non-overlapping A/B
roofline: latency-bound throughout — effective HBM stays at a few percent of nameplate, and after the win ~3/4 of the largest case is a size-independent fixed cost (fit of wall against bytes across an 8x size spread)
verified_on: 2026-08-17
last_seen: 2026-08-17
confirms_cited: 0
confirms_blind: 0
losses: 0
attempts: 0
---
# Fuse the quant passes behind a tag-slot grid barrier
- lever: A per-tensor quant that reduces a scale and then applies it does not need three dispatches — one kernel can do reset + absmax + apply if the grid barrier between the two phases is cheap; the fusion, not the body, is where the win is.
- apply: Make the barrier's cost the NUMBER of serialized cross-die round trips, not the instruction count: each block publishes (tag<<32 | scale bits) in ONE 64-bit relaxed agent-scope store into a per-block slot array, and every block loads the whole slot array coalesced (one wave, 8 bytes/lane), ballots arrival and reduces the scale itself with DPP wave ops. 2 round trips instead of ~6, and the tag doubles as arrival flag so it self-resets under graph replay. Let out-of-range lanes re-read slot 0 rather than branching to an identity, which keeps the ballot exact and the poll branch-free. Size the grid so the slot array is one coalesced load (~64 blocks here); when the flat element mapping would exceed that, widen the WORKGROUP rather than the per-lane vector.
- stack: total 1.73x weighted (director-verified, isolated vs frozen baseline) = three landings compounded
  - 1. fuse the three passes behind the tag-slot barrier — 1.55x standalone (round 1, verified) — carries the win
  - 2. conditional 128-bit-per-lane load, applied only when the grid stays under the barrier's coalesced-slot limit — +~2.7% paired on top of (1) (round 1 integrate, verified); applying it unconditionally was a net LOSS
  - 3. make the poll body O(1) — one coalesced slot load + arrival ballot + DPP reduce, no loop scaffolding — +6.2% paired on top of (1,2) (round 2, verified)
  - note: attribution is incremental in landing order; (2) and (3) were never isolated against the unfused baseline alone.
- verify: dispatch count per call drops to one in the trace; then decide accept/reject ONLY by unprofiled paired alternation of the two arms inside one thermal window (several full runs per leg), and read the min-duration floor separately from the mean — a patch that moves the mean while leaving the floor identical bought round-trip/ramp latency, not issued instructions.
- pitfall: fused candidate landed well below the unfused form -> the textbook protocol (atomicMax accumulator + padded arrival counter + last-arriver publish + generation spin) is ~6 serialized round trips -> replace with the one-store tag-slot publish above.
- pitfall: profiler durations ranked the arms backwards (the faster arm profiles slower) -> a grid-wide spin barrier makes kernel duration equal the LAST block's arrival, so it absorbs profiler-injected jitter -> trust the trace only for dispatch count, geometry, register/LDS/scratch and the min duration.
- pitfall: finer-grained per-wave barrier slots regressed heavily -> poll traffic scales as pollers x slots -> exactly one publisher and one polling wave per block.
- pitfall: separating the scale accumulator and the arrival counter onto different cache lines was slower, and explicit release/agent atomics lost to a plain atomic max plus a threadfence -> ordering was never the cost and L2 merges same-line atomic traffic -> keep them together; take the scale from atomicMax's RETURN value and use atomicInc's hardware wrap instead of tail re-reads and plain stores into the hammered line (that epilogue alone moved its direction from ~1.005x to ~1.109x).
- pitfall: a recorded patch had zero hunks for the file it claimed to change, and a later, correctly measured direction failed to apply at all -> the workspace copy has no VCS metadata so the diff was taken against an unrelated enclosing repo, and baseline-relative diffs rebase-collide when two directions rewrite the same region -> generate with a plain unified diff against the frozen baseline, dry-run the apply, and rebase onto the round's subject commit before submitting.
- caution: also verify what the harness actually times before funding a host-runtime direction — under captured-graph replay the launcher, dtype dispatch and argument marshalling all run once at capture and contribute exactly zero, so launcher memoization measured 1.000x here.
- caution: also verify grid size from BOTH ends before treating it as a free knob: raising block count regressed (cost tracks the shared atomic word, not grid fill) and collapsing the small case into a single workgroup so the barrier degenerates to a local sync also regressed — one CU's L2 port cannot absorb the traffic the barrier was saving.
- source: run kernel_20_geak_0811_2h, kb-clean lane, 2026-08-17; director-validated 1.73x weighted, correctness 3/3 pass
