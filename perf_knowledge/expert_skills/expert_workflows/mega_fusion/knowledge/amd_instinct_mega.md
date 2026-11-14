# AMD Instinct: what mode=mega adds to `amd_instinct.md`

Read `amd_instinct.md` (shared kernel_workflow knowledge) first; this adds what multi-rank fused kernels need.

On `gfx950`, read `gfx950_lowering.md` as well before you write or trust any synchronization,
occupancy, or residency argument. It carries what the emitted ISA does as opposed to what the source
says — the workgroup barrier does not drain `vmcnt`, the profiler's VGPR count is half the real one,
an agent-scope release is a cross-die writeback — and each of those reads as the benign answer at
source level.

## 3b. The die/cache hierarchy is not flat — synchronize at the narrowest level that is correct

An MI-series accelerator is **multiple dies (XCDs) under one device id**, each with its **own L2**,
backed by a shared last-level cache (Infinity Cache) and then HBM. Two consequences that a
CUDA-shaped mental model gets wrong, and that matter most in exactly the persistent/fused kernels
this workflow builds:

- **A "device-wide" sync is more expensive here than the equivalent on a monolithic part**, because
  the coherence point for blocks on different dies is below their L2s. Published AMD megakernel work
  (Fleet, and the kog follow-on) is organized almost entirely around this: sync *hierarchically* —
  intra-CU, then intra-die, then cross-die — and pay the wide barrier only for the edges whose true
  scope really is device-wide. `tile_task_graph.md`'s edge-scope column is where you record which
  ones those are; this is the hardware reason that column earns its keep. Independent reports note
  that AMD sync cost is a materially larger fraction of a fused kernel's time than on NVIDIA parts,
  so an overlap design ported one-for-one from an NVIDIA paper will underperform for this reason
  before any of its ideas are wrong.
- **The same line can be resident in several L2s at once**, so a producer/consumer pair split across
  dies pays a real cost that is invisible in a per-CU model, and keeping a working set at the
  highest cache level that still covers its consumers is a lever. Unlike LDS, L2/LLC are not
  directly programmable — you steer them with cache-policy bits and non-temporal hints (see
  `distributed_fusion.md` Lever 6 for the gfx95x bit layout), which makes this a *measure it* lever
  rather than a *design it* one.

Do not carry a die count from the table above; it varies by SKU and by the partitioning mode the box
is booted in (`rocm-smi --showcomputepartition`). NPS/compute-partition mode changes what "the
device" means for both grid sizing and coherence, so read it before you reason about either.
