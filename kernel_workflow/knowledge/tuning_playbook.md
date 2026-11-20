# Tuning Playbook — the `tune` specialty

How a `tune` engineer works. engineer.md still governs everything else (rules, gpu_lock, correctness
before benchmark, `focus_files`, outputs, `measurement_valid`). This file replaces engineer.md's
"implement → benchmark → iterate a few variations" with a **measured configuration search**.

Why a separate specialty: many kernels leave real speed in their configuration — tile shape, waves
per workgroup, pipeline depth, MFMA instruction shape / `kpack`, unroll, split factors, and above all
*one config serving every shape*. Specialists change the code and keep its old config; a config
chosen for the old code is often wrong for the new one. Observed on GEAK runs: retuning right after a
structural commit found +6.8% on a kernel whose previous tune had been made obsolete by that commit
(`BLOCK_M 128→64`, `NUM_UNROLL_CACHE 8→1`); a per-shape branch for a GPU-unfilled small case gave
1.40× on that case alone.

## What you change
- **Primary:** values of existing parameters and per-shape config dispatch (split or move a shape
  threshold, give each regime its own config). This is where you spend most of the round.
- **Allowed in service of the search:** small code changes that let a better config exist — expose a
  hard-coded value as a knob, add a per-shape dispatch branch, enable an existing kernel variant a
  config needs, fix a launch-bound/occupancy hint. Keep them minimal and inside `focus_files`; a
  rewrite of the algorithm or memory path belongs to the other specialties (say so in `notes`).
- **Never:** external tuned artifacts (aiter/hipBLASLt/CK tuned CSVs, TunableOp, library heuristics) —
  the win must live in the kernel's own source; changed numerics (dtype, accumulation, scaling).

## Workflow
1. **Scope (≤10 min).** Read the hot kernel + wrapper. Inventory every parameter: name, location,
   current value per shape regime, legal values, and the hardware constraints that bind it (LDS bytes
   per workgroup, accumulator VGPRs/AGPRs, divisibility by the MFMA/WMMA shape and by the problem
   dims, workgroups vs CU count — see the hardware reference). Map each scored case to the config
   branch it takes today. Nothing tunable and nothing worth exposing → return `status: partial` at
   once with a `notes` line saying so.
2. **Baseline.** Run the COMMANDMENT benchmark once via gpu_lock on the untouched workspace; record
   per-case latency and the run-to-run noise you observe.
3. **Prune, then seed.** Drop everything the hardware constraints rule out BEFORE compiling, then
   write down a few reasoned seeds per shape regime (small / GPU-unfilled regimes usually want
   smaller tiles, more split, more waves in flight; large compute-bound regimes want MFMA-friendly
   large tiles and enough workgroups to fill the CUs). Seeds first; random fill only if time remains.
4. **Screen cheaply.** A scratch script that imports the kernel once and times the scored cases
   in-process (same inputs, warmup and cache flush as the harness) is far cheaper than the full
   harness; check correctness against the reference for EVERY variant — configs that compile can
   still compute wrong results. Vary one axis at a time, combine the per-axis winners, then refine
   around the best joint configs.
5. **Choose per shape.** Encode the best config per regime as config values / dispatch branches. Do
   not accept a global change that regresses any scored case by more than noise when a per-shape
   branch would avoid it.
6. **Prove engagement.** Show that the config you chose is the one that actually runs for each scored
   case, in the SAME module the harness imports (print/assert the selected config once per case in a
   scratch run). A stale overlay or module copy silently times the old kernel; when many configs come
   out within noise of each other, suspect that (or an incompletely keyed JIT cache — move it aside,
   never `rm`) before believing the numbers.
7. **Confirm with the real instrument.** Run COMMANDMENT correctness, then the COMMANDMENT benchmark
   at least twice interleaved with the untouched baseline (A B A B). Screen winners are optimistic
   (winner's curse); only the harness number counts. Report per-case speedup vs `baseline_per_case`,
   the geomean and, for a workload-aligned COMMANDMENT, `speedup_weighted` (the PRIMARY number).
   self_monitoring's "3 benchmarks within 1%" stop applies to these confirmations, not to screening.
8. **Save `best_patch.diff`** the moment a harness result clears the floor named in your prompt, and
   re-save whenever best improves. Aim to return within ~45 min of wall time.

## Report
In `strategy`: the final config / dispatch per regime (e.g. `M<=32: WM=1,WN=4 (was 2,2); M>2048: RN=2`).
In `strategies_tried`: the axis/config families screened, with how many variants each. In `notes`:
the knob inventory, which axes mattered per regime and which were flat, configs that failed
correctness or spilled, and knobs that would need a structural change to expose — a flat space is a
useful finding for the TechLead. `report.md` adds the search-space size before/after pruning and the
per-case table (baseline / tuned / speedup).

## Using the tuning skillset
When `KERNEL_KNOWLEDGE_DIR` is non-empty, `expert_skills/tuning/tuning-core/` (`SKILL.md`,
`search_strategy.md`, `measurement.md`, `correctness_gates.md`, `engagement_verification.md`) and the
`tuning-<language>/SKILL.md` for the kernel's current language hold the details behind steps 3–7, and
`operators/<KK_OPERATOR>/tuning.md` the known knobs for this operator. They are written for tuning
deployments inside frameworks; here "deploy" means the edited source in `KERNEL_PATH` is what the
harness runs. Facts and how-to, never decisions — measurement decides.
