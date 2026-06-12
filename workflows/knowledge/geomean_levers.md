# Geomean Levers — How to Beat the Wall-Clock Floor

The headline metric is **geometric mean of per-case speedup** across the benchmark's test cases.
Geomean is dominated by the WORST relative speedups, not the best. A kernel that is 70x on big
shapes but only 10x on small shapes has a geomean closer to the small number. This file is the
playbook for the directions that move geomean the most once the raw kernel is already fast.

## The core insight: kernel-fast ≠ wall-clock-fast

When a brute-force kernel gets a warp-cooperative + template rewrite, the GPU compute time often
drops 40–60x. But the measured per-call latency does NOT drop that much on small/medium shapes,
because the wall clock is now floored by **fixed per-call overhead** that is independent of problem
size:

- Python wrapper / dispatch / `torch.autograd.Function` machinery
- Host-side tensor reshaping: `.transpose().contiguous()`, `.clone()`, layout fixups
- Output allocation and zero-init (`torch.zeros`, `new_zeros`)
- Kernel **launch latency** × number of kernel dispatches per call
- Multiple sub-kernels per call (e.g. a separate init kernel + main kernel + copy kernel)

**Diagnostic**: if several test cases of very different problem sizes all land within ~2–3x of
each other in latency (e.g. shape_0 ≈ 0.12 ms while it computes 100x less than shape_3 ≈ 0.8 ms),
the small/medium cases are overhead-bound. That is exactly where geomean is being lost.

## Lever 1 — Collapse the dispatch count (highest impact on small shapes)

Count how many GPU kernels launch per call (profiler `top_kernels`, or `rocprof --stats`). Each
launch is ~5–15 us of fixed latency. If a call does init + main + copy + cast = 4 dispatches, that
is 4× launch latency on every single call regardless of size.

Directions:
- **Fuse all sub-kernels into one** `__global__`. Do init/cast/copy inside the main kernel.
- Remove "elementwise"/"vectorized_elementwise"/"distribution" helper kernels that exist only to
  fill or cast a buffer — fold that work into the producing kernel or eliminate it.
- Avoid launching a separate kernel for a `.contiguous()` / `.transpose()` — make the main kernel
  read/write the desired layout directly (see Lever 3).

## Lever 2 — Kill host-side overhead (Host/Runtime specialist's home turf)

See `wrapper_optimization.md` for the patterns. The big ones:
- `torch.empty` instead of `torch.zeros`/`new_zeros` for outputs the kernel fully writes.
- Drop unused scratch/output allocations entirely (e.g. a `dist2` buffer the caller never reads).
- Bypass `torch.autograd.Function.apply` with a plain `@torch.no_grad()` function.
- Skip redundant `.contiguous()` when the caller already guarantees contiguity.
- Skip `CHECK_*` macros in the C++ binding on the hot path.

## Lever 3 — Native layouts: make the kernel emit the final layout

The single biggest host cost is usually a post-kernel `.transpose(2,1).contiguous()` and/or a
pre-kernel `.transpose().contiguous()` on the inputs. These are full memory-traffic passes on the
host side that can cost more than the kernel itself on large shapes.

Directions:
- **Output directly in the caller's expected layout** (e.g. write `idx[b, k, m]` with the right
  strides) so the wrapper returns the kernel output as-is, no transpose.
- **Read inputs in their native layout** via a compile-time `template <bool TRANSPOSED>` so the
  transposed code path reads SoA `[b, c, i]` directly instead of forcing a host transpose. Use
  `if constexpr (TRANSPOSED)` inside the kernel — zero runtime dispatch cost. (Respect the hipify
  safety rules in `hip_optimization.md`: dispatch templates via a template function, never a macro
  with `<<<>>>` in an if/else.)

## Lever 4 — Attack the SLOWEST per-case, not the fastest

Because geomean weights the worst case heavily, find the lowest-speedup row in the per-case table
and target it specifically:
- If the worst case is the largest-N / highest-k shape, it is likely compute or VGPR-pressure bound
  (e.g. K=10 keeps a 10-wide register array per lane → spills, low occupancy). Consider LDS for the
  top-K merge, or a different K specialization.
- If the worst cases are the smallest shapes, they are overhead-bound → Levers 1–3.
- A self-query variant (center == data) may hit a different path — make sure it uses the fast kernel
  too, not a fallback.

## Lever 5 — Persistent / grid-right kernels for small problems

Tiny shapes underfill the device's CUs (304 on MI300X/MI325X, 228 on MI300A, 256 on MI350/MI355 —
detect with `rocminfo`, see `amd_instinct.md`). A grid of a few blocks wastes launch latency relative to work.
- Use **persistent threads** (launch ~#CU blocks, loop over work items) to amortize launch.
- Or batch multiple logical calls into one launch when the harness issues several in a row.

## Lever 6 — CUDA/HIP graph / stream capture (collapses the launch-overhead floor)

When the same sequence of launches repeats every iteration, capture it once into a HIP/CUDA graph
and replay. This removes the per-call CPU launch + Python dispatch overhead and is the primary lever
for the overhead-bound / floor-dominated regime.

**This applies to single-kernel benchmarks too, not just e2e.** The per-case benchmark harness times
MANY repeated calls of the same op, so it is a repeated-call workload by construction — exactly the
pattern graph capture is for. Do it at the **wrapper layer** (capture the full op: layout fixups +
launch + any reductions), not at the C++ `<<<>>>` launcher (launcher-level capture cannot reach the
Python/dispatch overhead that forms most of the floor, and is typically a dead-end). **Gate it on
measured replay benefit**: build the graph once, compare replayed latency vs eager per shape, and
use replay only where it actually wins (so it never regresses shapes that don't benefit).

When the geomean is floor-dominated, this is frequently the single largest geomean win available —
it lifts EVERY floored case at once — so treat it as a first-class direction, not a last resort.

## CRITICAL: the floor-dominated-signal trap (read this every round)

When the machine is fast (or lightly loaded), the small/medium cases sit at the fixed launch
overhead floor (e.g. every small shape ≈ 0.012 ms regardless of size). In that regime the **geomean
becomes floor-dominated**: most cases are already at the floor, so the geomean barely moves no matter
how much faster you make the *kernel*. This silently mis-steers the optimizer — it gets rewarded for
overhead work it has already won, and **under-rewarded for real kernel-compute gains on the large,
compute-bound cases** — so it converges to a mediocre kernel and stops too early.

How to avoid it — attack BOTH ends (the floor is improvable, not "done"):
1. **Detect the floor, then COLLAPSE it.** If several cases of very different problem sizes share
   nearly the same latency, that latency is the floor. The floor is NOT unimprovable: under the
   repeated-call benchmark harness it is directly attackable with wrapper-level HIP-graph
   capture/replay (Lever 6). When the geomean is floor-dominated (most cases sit at the floor),
   collapsing the floor is the single highest-impact direction — it lifts every floored case at once
   — so dispatch a `host_runtime` graph-capture direction to attack it BEFORE concluding those cases
   are done. A floored case is only truly "done" once the floor itself has been attacked.
2. **In parallel, drive down the compute-bound cases' ABSOLUTE latency.** Identify the case(s) whose
   latency is well ABOVE the floor (the largest-N / highest-k shapes). Drive THEIR absolute
   milliseconds down. A direction that halves the worst compute-bound case is a win even if the
   floor-dominated geomean barely changes. (Do this alongside Lever 6, not instead of it.)
3. **Do not declare victory on a flat geomean** while the worst compute-bound case is still many×
   the floor — that means the kernel still has compute headroom. Keep pushing kernel efficiency
   (better warp-cooperative scan, lower VGPR/occupancy, LDS merge, ILP/sorting-network) until the
   compute-bound case approaches the floor too, or returns clearly diminish.
4. A genuinely excellent kernel pulls EVERY shape down to (or near) the launch floor. If your large
   shape is still 3–5× the floor, the kernel is not done — regardless of what the geomean says.

## Priority when chasing geomean
1. Measure dispatch count + per-case table. Identify which cases are overhead-bound.
2. Lever 1 (fuse dispatches) + Lever 3 (native layout) usually give the largest geomean jump once
   the kernel compute is already fast.
3. Lever 2 (host) cleans up the remainder.
4. Lever 4 then squeezes the single worst case.
5. Lever 5/6 for the small-shape / repeated-call floor. When the geomean is floor-dominated (most
   cases already at the floor), promote Lever 6 (graph capture) to the TOP priority — it is the only
   lever that moves the cases carrying the geomean, so do it before chasing compute-bound outliers.

Do NOT stop optimizing just because GPU kernel time is small — at that point the overhead IS the
bottleneck, and these levers are where the remaining 1.5–3x of geomean lives.
