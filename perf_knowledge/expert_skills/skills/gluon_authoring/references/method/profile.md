# Profile — reading rocprofv3 / rocprof-compute / static ISA on gfx950

Advisory reading rules for the profile evidence. GEAK's `kernel_workflow/scripts/profile_kernel.sh
<gpu_id> "<cmd>" <out>` is the profiler entry (runs under `gpu_lock.sh`); general interpretation is in
`kernel_workflow/knowledge/profiling_guide.md`, tool usage in `kernel_workflow/scripts/kernel_tools/CARD.md`.
This page holds the gfx950 / Triton-kernel specific traps. `$KT` = `kernel_workflow/scripts/kernel_tools`.

### Wrapper conventions

- **Never call `rocprofv3` bare; use `$KT/rocprofv3_safe.sh --kernel <regex> --out <dir> [--pmc "A B C"]
  [--timeout 150] -- <cmd...>` under `gpu_lock.sh`.**
- **`HIP_VISIBLE_DEVICES` is re-indexed inside `ROCR_VISIBLE_DEVICES`**, so exporting the same physical
  index in both leaves the profiler child with no device: on gfx950 / rocprofv3 1.3.5 the app failed at
  once (it had earlier been recorded as a "TCC counters hang rocprofv3", and every derived figure from that
  run was withdrawn). For a profiler child set one of them, not both with the same index; the safe wrapper
  resolves HIP-inside-ROCR for the child. GPU-checked.
- **Suggested: a hard timeout (`PMC_TIMEOUT=150` s); exit 124 = timed out** → record a degraded layer, never a zero. An
  unschedulable counter group may replay, SIGABRT, or hang holding the GPU flock (version-dependent;
  **Overrides KB** for the rocprofv3 releases used with this pack on gfx950: `perf_knowledge/profiling/rocprofv3_counters.md`
  "rocprofv3 **replays the kernel**" is one of these outcomes, not the rule; basis: the outcomes recorded in
  `kernel_workflow/scripts/kernel_tools/rocprofv3_safe.sh` GUARD 1, which differ by rocprofv3 release).
  Do not apply this timeout to e2e serving captures.
- **Filter with `--kernel-include-regex`**: collection is limited to the matching kernels, and
  unfiltered rows include torch/CK helpers you would otherwise average in.

### Execution locus

**Kernel in a separate container: the profiler must run in that container** (`kernel_workflow/scripts/kernel_tools/locus.sh`,
`TILE_KERNEL_CONTAINER`, shared workdir via `TILE_KERNEL_CONTAINER_WORKDIR`). A host-side profiler sees
no in-container JIT kernel (`sol_pmc_no_kernel_rows`, analyze-blind, ATT `code:null`) — a fixable
mis-configuration, not a blind mode. `capture.sh` exits 4 when a container is configured but the
profiler cannot run in it.

## Profiler-capability preflight (verify ONCE — do not assume PMC is visible)

**rocprofv3 can capture 0 dispatches of a JIT-launched Triton/Gluon kernel** (only torch helpers),
silently yielding no counters. Check once per box:

```bash
bash kernel_workflow/scripts/gpu_lock.sh <gpu> rocprofv3 --kernel-trace -f csv -d /tmp/kt -- python bench.py
grep -c <your_kernel_name> /tmp/kt/*/*kernel_trace.csv      # 0 => not observed
```

- **0 because rocprofv3 aborted is a different failure — check the log first.** When torch and the
  rocprofv3 tool each load their own `libamd_comgr.so.3` (it carries the SPIR-V translator), the process
  aborts with `Option 'spirv-expand-step' registered more than once` / `LLVM ERROR` and then hangs (it
  ignores SIGTERM: run under `timeout -s KILL`). Observed in a ROCm TheRock 10.0 image (rocprofiler-sdk
  1.3.5): one copy came through the rocprofv3 tool's RPATH (`_rocm_sdk_devel`), the other through torch's
  `rocm_sdk` preload (`_rocm_sdk_core`). It hits every rocprofv3 mode (`--kernel-trace`, `--pmc`, `--att`),
  so the grep above reads 0 and a wrapper may report a clean run. Fix: make the two paths one file — a
  hard link works only inside a running container (after `docker build` the copies differ again), so use
  a symlink in an image; `LD_LIBRARY_PATH` does not help (RPATH wins) and `LD_PRELOAD` of the core copy
  fails with rocprofiler error 16. Route to the library fix, not to the ISA-only path below.
- `>0` proves trace visibility only; validate counters and the ATT decoder separately.
- **0 dispatches → native ISA path**: `$KT/dump_ir.sh` → `.amdgcn`/`.s` → `$KT/asm_loop_audit.py`
  (VGPR/AGPR/spill from the `.amdgcn`) + the floor probe (`### Rule: floor probe (quantify overlap
  headroom exactly)`). The Triton cache already holds the ISA.
- Also blind: kernel launched inside a CUDA/HIP graph replay or from a `dlopen`'d external `.so` →
  timing at the graph boundary + ISA. `rocprof-compute` rc=4 / "no workload dir" (one cause: missing
  Python deps `plotext` / `astunparse` — `python3 -c "import plotext, astunparse"`) → degrade to
  `rocprofv3 --pmc` or timing+ISA (an argv JSON config through `docker exec` is one cause:
  `benchmark-hygiene.md ## Profiler output hygiene (do not pollute the repo)`).
- Degrade order when a layer fails: rocprof-compute analyze → `rocprofv3 --pmc` (`parse_pmc.py`) → ATT
  on a smaller shape → native ISA + floor probe + boundary timing (also for a grid too large to replay).

### Rule: pipe busy is not a fraction of peak

**`MfmaUtil` / MFMA busy-cycles = fraction of cycles the pipe was occupied, not fraction of peak
throughput** (the instruction mix can run below the dtype's peak rate per busy cycle; normalization
is per SIMD or per CU depending on source). Before calling a kernel "at the MFMA ceiling", compute the
achieved rate (`gluon_profile.py --flops`) against this dtype/SKU peak (`hw_budget.py
--measured-tflops`). High busy + low achieved fraction = schedule / instruction-rate problem.

## 3.1 Evidence per round — the four dials

`scripts/gluon_profile.py` collects these per call. Only the stack-specific traps are listed.

**A — occupancy and registers**
- **A1: `asm_loop_audit.py` reports waves/SIMD; bandwidth residency is workgroups/CU.** A workgroup is
  indivisible; they coincide only at `num_warps == 4` (SIMDs/CU), so a VGPR cut that raises waves/SIMD
  can leave workgroups/CU, and bandwidth, unchanged. Carry `num_warps` as a column.
- **LLVM's `; Occupancy: N` comment is the register term only — no LDS term.** On dynamic-LDS Gluon
  kernels it overstated occupancy (gfx950 / Triton 3.8.0); use `min(reg, LDS)` (`$KT/amd_occupancy.py`,
  `$KT/probe.py measure --dir <ir>`; formulas: `../hardware/planning-constants.md`).
- **A2:** at `num_warps >= 8` `vgpr_count` cannot exceed 256, so a VGPR gate "passes" while the build
  spills — read `.vgpr_spill_count` from the same row (rule:
  `../hardware/planning-constants.md ## waves/SIMD is not workgroups/CU — convert before spending a round on it`).
- **A3: LDS bytes come from the Triton cache metadata `shared` (`meta_*.json` beside the dump).** The
  kernel descriptor's `group_segment_fixed_size` and the `.amdgcn` `LDSByteSize` are **structurally 0 for
  Triton kernels** (LDS is dynamic; the `.amdgcn` fields read 0 for a kernel with shared memory,
  compile-checked on Triton 3.8.0 / gfx950) — a 0 there is not evidence. Profiler columns are tool-dependent: rocprofv3
  1.3.5's kernel-trace CSV `LDS_Block_Size` reports 0 for every dynamic-LDS kernel (static LDS only), while
  rocprof-compute 3.11.0 block `7.1.8` reported the dynamic allocation (= `shared`) on gfx950 — cross-check
  any profiler LDS column against `shared`.

**B — instruction stream**
- B1/B2: matrix-issue intervals by preceding op class (`mfma_efficiency.py`, ATT); LDS vs global feed
  split. Grouping is observational, not a stall attribution.
- B3: per-opcode hot-loop histogram (`asm_loop_audit.py <f>.s --opcodes N`) diffed per round; keep
  `s_waitcnt` in it (moves when only the schedule moved).
- **`ds_read_b128` steady issue interval (gfx950, all four SIMDs reading): 16 cycles = conflict-free; 32 / 64 =
  2- / 4-way bank conflict.** Load-dependent (one wave per CU: 8 conflict-free, 2-way hidden; SIMD pairs share
  read throughput), so read the ratio to a conflict-free baseline under the same load and confirm with
  `SQ_LDS_BANK_CONFLICT` (GPU-measured, gfx950). gfx942's conflict-free interval is unverified (16 or 32;
  `../hardware/cdna3-gfx942.md`).
- B4: `amd_isa.py --arch <gfx> facts <V_MFMA_...>` before picking an MFMA tile (equal MAC/cycle shapes can differ 4x in
  accumulator GPRs/lane).

**C — memory side**
- C0: check the roofline applies (`hw_budget.py --grid --footprint-mb --dispatches`): fewer WGs than
  CUs, working set inside the LLC (re-runs are then served from MALL and can beat a floor built from
  measured HBM bandwidth), or several dispatches in the timed region void it.
- **C1 counters: `FETCH_SIZE WRITE_SIZE TCC_HIT_sum TCC_MISS_sum` (+ `TCC_EA0_*`), via
  `rocprofv3_safe.sh` → `parse_pmc.py`.** They sample the L2/fabric boundary (cannot see LLC vs HBM);
  routes agree on a pure stream and diverge several-fold on real kernels (miss counting misses
  partial-line stores, write-allocate, atomics → reads low). Cross-check at least two routes.
- **The counters-per-pass limit is per hardware block, not a counter count** (gfx950, rocprofv3 1.3.5):
  `FETCH_SIZE` + `WRITE_SIZE` in one pass failed (error 38, "Request exceeds the capabilities of the
  hardware") because each expands to several TCC request counters, while `TCC_HIT_sum TCC_MISS_sum
  TCC_EA0_RDREQ_sum TCC_EA0_WRREQ_sum` collected together and up to 8 raw SQ counters fit (9 failed).
  `parse_pmc.py`'s groups are split to fit; on error 38 / abort split the list (`collect_pmc_group()`).
  gfx942 limits were not checked.
- C2/C3: `hw_budget.py` refuses %-of-ceiling until the denominator is probed (`mem_bw_probe.py`, a dense
  MFMA loop → `--measured-tflops lo,hi`). Declare the FLOP engine (`--flops-engine mfma|valu`): pricing
  a reduction/norm at the MFMA rate puts the compute floor 1–2 orders too low.
- C4: the floor probe is what confirms memory as the bound.

**D — timing / correctness**: screening via `kernel_workflow/scripts/kernel_tools/ab_bench.py` (interleaved, `--control`,
`--permute`); acceptance is GEAK's. After adding a default-off `constexpr`, re-hash the shipped stream
(`benchmark-hygiene.md ### A knob that is off still has to prove the shipped stream did not move`).

Record the toolchain identity with every number; `TRITON_HIP_USE_COEXEC_SCHEDULER` is read nowhere on
3.8.0 (`../tile-programming/compiler-contract.md ## Toolchain identity`).

## Bound classification -> primary metric

| bound | read | lever family |
| --- | --- | --- |
| MFMA-issue | `MfmaUtil` + achieved rate vs peak | pipeline / slicing |
| VALU | `VALUBusy` (not `VALUUtilization`) | decompose the VALU, then reduce |
| dependency (latency C1) | dependency-wait / Total (waiting on memory of any kind); `VALUBusy` and `MfmaUtil` both low | more in flight (prefetch, occupancy), shorten the chain |
| occupancy (latency C2) | achieved occupancy + SPI limiter (not issue-wait: issue-wait is pipe contention) | registers / LDS / grid fill |
| memory | C1 vs probed ceiling, only after C4 | memory path |
| LDS | `ds_read` interval, `LDSBankConflict` | padding / swizzle |

**Dependency wait vs issue wait have opposite fixes and flip with tile size** — one kernel read mostly
dependency wait at the small tile and mostly issue wait at the large one. Divide each by Total wave cycles
(never subtract them). Re-read after every tile / `num_warps` change. Decision tree and cutoffs:
`../hardware/bound-class-signals.md ## Bound-class decision tree — the AMD discriminator table`;
rocprof-compute block IDs: same file, `## rocprof-compute block-ID cheat-sheet (when a rocprof-compute
report is available)`. `--roof-only` on gfx95x needs rocprof-compute >= 3.6.0.

## Derived metrics (direct `rocprofv3 --pmc` read)

CDNA (gfx9) derived names, one pass beside the memory counters:

```text
MfmaUtil         matrix-core busy: fraction of ALL cycles MFMA worked
VALUBusy         VALU busy/throughput: fraction of ALL cycles VALU worked  <- classify on this
VALUUtilization  lane occupancy DURING active VALU cycles (not throughput)
LdsUtil  LDSBankConflict  OccupancyPercent
MemUnitStalled   % cycles the memory unit stalled (~0 => not memory-bound)
```

Memory: `TCC_EA0_RDREQ_DRAM_sum,TCP_TCC_READ_REQ_sum`. Raw SPI counter names are build-specific —
probe per box (`rocprofv3 -L`). Whether an unresolvable name fails the whole job is build-specific:
rocprofv3 1.3.5 on gfx950 dropped an unknown name with a warning and collected the rest — check the
warning, not just the exit code.

### Rule: read the busy/throughput counter, not the lane-activity one

**Classify a compute bound on `VALUBusy` / `MfmaUtil`, not `VALUUtilization`** (**Overrides KB** for
rocprofiler-sdk 1.3.5, gfx9 definitions: the "VALU utilization" row of
`perf_knowledge/profiling/rocprofv3_counters.md`, which maps it to `VALUUtilization`; basis: that release's
counter descriptions (`share/rocprofiler-sdk/config.yaml`), where `VALUUtilization` is "the percentage of
active vector ALU threads in a wave" (lane activity) and `VALUBusy` is "the percentage of GPUTime vector ALU
instructions are processed"). `VALUUtilization` ~100%
with `VALUBusy` and `MfmaUtil` both well under 100% and `MemUnitStalled` ~0 = dependency/latency-bound
(C1), not at a ceiling. Corollary: cutting instructions of a unit whose `*Busy` is low is expected neutral; the
productive levers are a shorter critical path or more occupancy. Confirm a 100% reading by removing
some of that unit's work and checking time moves.

### Rule: floor probe (quantify overlap headroom exactly)

Delete the whole dependent stage (e.g. the softmax body), keep it compiling (correctness void), time it:
`exposed_fraction ≈ (full - floor) / full`. If the floor itself is far from peak, that stage is not
the wall. Cumulative constexpr gating (`PROBE>=1, >=2…`) gives per-stage time shares of a fused chain;
counters stay whole-kernel. Shares from components captured in isolation are an interval
(`isolated/sum_of_parts` .. `isolated/whole`; isolated parts can over-sum the whole), and Amdahl on them
over-predicts. On plain Triton the pipeliner re-schedules the probe, so trust only the dominant stage.

**Bound-the-win probe:** before building an overlap/prefetch lever, remove the dependency or barrier it
targets (accept wrong output) and time it; ceiling ≈ current ⇒ the lever is worth ~0. A probe bounds a
win; it never declares a ceiling by itself.

### Reducing compute-class VALU (fold scalars off the tile)

When `VALUBusy` binds, histogram the loop's VALU into compute (`v_exp`/`v_fma`), layout-convert
(`v_perm`/`permlane*`/`v_cndmask`/`ds_bpermute`), register-shuffle (`v_accvgpr_read`) and
integer/address classes (SOL `2.1.1 VALU IOPs` ≫ `2.1.0 VALU FLOPs` = instruction-bound on address
math). For compute class, fold per-element scalars onto the cheapest upstream carrier:
`exp(qk*s - lse)` → `exp2(qk*(s*log2e) - lse*log2e)` (one FMA + `v_exp_f32`), or pre-scale `Q` at load
(`../workloads/attention.md ## Online softmax`); seed the accumulator with `-bias`. Verify in asm the
per-element op is gone; expected to be neutral on a kernel that is not VALU-bound.

### Raising exp throughput by partial FMA emulation (when the exp unit is the bound)

**The signal for it: the transcendental unit's busy counter saturates while MFMA/VALU idle.** Compute a
fraction of the `2^x` on the FMA pipe (Cody-Waite: exponent bits for `floor(x)`, degree-3 Horner
polynomial for the fraction, ~1 BF16 ULP) and leave the rest on `v_exp`. Emulating all of them spilled in the cases recorded;
tune the fraction by the measured throughput ratio, and keep waves/CU unchanged
(`../tile-programming/slicing.md ## Occupancy budget (P8)`). On a latency- or `ds_read`-bound kernel
the exps are likely already hidden and this is expected to be neutral-to-negative. Inferred, not measured on gfx950: the exp
unit binds more readily on faster-MMA parts (gfx1250) than on gfx950, so prove the bound first. A
loss here closes this recipe, not cheaper fast-exp forms (bit-construction exp for fp8 P, max-free
softmax): `../workloads/attention.md ## Online softmax (the recurrence to preserve)`.

### Rule: low MfmaUtil has two distinct tile-size causes — read the geometry to tell which

- **Under-filled:** `tile_dim < warps_on_axis × instr_extent` → half-empty MFMAs; the remedies are growing the
  tile or not partitioning that axis (worth a small spill).
- **Under-amortized:** tile filled, but a serial per-row stage (softmax/rescale) dominates → `VALUBusy`
  binds; a bigger M tile amortizes it. The plain autotuned tile is a free oracle.
- **Over-coverage** (`warps_per_cta[d] * instr_shape[d] > extent`): redundant traffic on an input tile,
  a work multiplier on the output tile. The ratio is a coarse screen, not a price (one
  case: the ISA showed several times the work per block the ratio predicted). ISA counts are **per warp**; multiply by warps
  before comparing to per-block quantities.

On a transcribed Gluon anchor (literal layouts) a tile change re-recovers the whole layout set, so it is
not a free knob; layouts built parametrically from the tile can make it one.

### Rule: equal registers+occupancy but lower MfmaUtil => schedule/overlap gap

**Take register/AGPR counts from the static `.amdgcn` (`.vgpr_count`, `.agpr_count`,
`.amdhsa_accum_offset`), not from the profiler: on gfx950 / rocprofv3 1.3.5 the `VGPR_Count` /
`Accum_VGPR_Count` columns did not match the descriptor (no fixed ratio; the Accum column was non-zero on
kernels with no AGPRs).** With
equal registers/`OccupancyPercent` vs plain but lower `MfmaUtil` and more full-drain `s_waitcnt
lgkmcnt(0)` (`asm_loop_audit.py`), the gap is the schedule (no `MemUnitStalled` saturation = not
bandwidth). Candidate repairs (`../tile-programming/pipeline.md`): register prefetch fits a
register-resident tile or a loop with few MFMAs per iteration; an authored LDS ring fits a compute-bound
LDS-staged tile; `warp_pipeline_stage` sits on top of either; re-injecting plain's pipeliner is diagnostic only (`recover.md`).
`num_stages` is not a Gluon lever on 3.8.0.

## rocprofv3 ATT (optional `--att`)

Run via `$KT/capture.sh <name> <out> --kernel <substr> --source kernel.py --dev <gpu> --cmd "<cmd>"`
(auto-widens and retries) or `rocprofv3_safe.sh --att`; read with `hotspot_analyzer.py
<out>/att/ui_output_agent_* --topk 15 --mode both --arch gfx950 --kernel <substr>` (without `--arch` /
`--kernel` the register/occupancy block reads unknown; for a Gluon kernel prefer `--mode asm` — source-line
mode put most stall on one inlined line), `att_opclass.py <dir>/code.json`, and
`mfma_efficiency.py "$(ls <dir>/se*_wv*.json | head -1)" <dir>/code.json` (one wave file: a glob that
matches several waves fails argument parsing). Pass an absolute `-d` to a bare rocprofv3: a relative one is
resolved against the app's cwd at exit. Checked on gfx950 / rocprofiler-sdk 1.3.5.

- **ATT decoder `librocprof-trace-decoder.so` is not vendored.** On ROCm >= 7.13 images it ships with
  rocprofv3 and is found without settings (`ls /opt/rocm/lib | grep trace-decoder`; a TheRock 10.0 image
  carried 0.2.0, matching rocprofiler-sdk 1.3.5). Only on older images install it separately and set
  `ROCPROF_ATT_LIBRARY_PATH` (rocprofv3 `--att-library-path`); the standalone ROCm/rocprof-trace-decoder
  releases are deprecated and older. Absent decoder = degrade (drop ATT), not failure. A loadable decoder
  does not mean ATT works under Triton: the duplicate-comgr abort (`## Profiler-capability preflight`)
  is the usual blocker.
- **`code.json` with `code: null` (traced CU caught no waves) → widen: `--att-target-cu 0
  --att-shader-engine-mask 0xF`.** On a small grid CU 0 may receive no workgroup at all and widening does
  not help: launch at least one WG per CU, or trace another `--att-target-cu`.
- **`mfma_efficiency.py` prints `ideal=UNKNOWN` unless `--ideal-cadence` is given**; take the per-shape
  issue interval from `../hardware/isa-mechanisms.md ## MFMA throughput model (planning cycles)` (gfx950
  intervals were ATT-measured).
- **Decoded wave record** (for manual reading): `[time, category, stall, duration, line]`; issue cycle =
  time + stall; MFMA appears as category 6 (VALU).
- Reorder-type levers were observed inert at ~1 wave/SIMD in the cases recorded; run-length (`deep_mfma_analysis.py`) and
  occupancy are the signals to read. Reference-free reading: `../hardware/bound-class-signals.md ## Reference-free ATT
  reading (no gold / vendor kernel at runtime)`.

### rocprof-compute per-round (aggregate SOL + memory chart + occupancy limiter)

`rocprof-compute` (via `profile_kernel.sh`); collection and analysis in the same locus. Non-redundant signal: memory block (HBM/MALL BW + L2 hit), `warp_state` (dependency /
issue wait), coalescing. A SOL failure on a PMC-capable box is usually wiring (container locus, argv
JSON config stripped on replay). `parse_pmc.py` keeps absent metrics null; raw counts cannot be
normalized into utilization.

## Asm hot-loop audit

`$KT/asm_loop_audit.py <variant>.s` (signals only) on both plain and Gluon: op-class histogram,
`s_waitcnt` relaxed (`cnt>0`) vs full-drain (`(0)`), barriers per iteration, `s_nop`.

### Reading the machine code of a JIT comparator you have no source for

1. Byte-scan the JIT cache for the dispatched kernel name — it yields a **candidate set** (one case: 16
   entries), not a selector; never tie-break by mtime (reads do not touch mtime; the dispatch used the
   older entry). Wrap `open`/`os.open` around one comparator call and record the path it opens.
2. **Scan for the ELF magic in every spelling: raw, `\x7f`, and MLIR-style `\7FELF` (uppercase hex, no
   `x`)** — `gpu.binary` payloads use the latter; scanning only `\x7f` reports "no ELF" silently. Take
   the image length from the section table (`e_shoff + e_shentsize*e_shnum`, extended by sections).
3. `llvm-objdump --mcpu=gfx950`: **AMDGPU output is mnemonic-first, address in a trailing `// ADDR:`
   comment**; a host-layout parser matches zero lines. Assert the parse is non-empty. Branch targets are
   symbol-relative. Execute nothing from the cache.
