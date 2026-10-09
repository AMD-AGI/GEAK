# Profile — the evidence every decision stands on

**What this stage decides.** Which resource binds (the bound class *and* the binding sub-resource),
and a **ranked, absolute-numbered bottleneck stack** the climb descends. Nothing else in the method
may be decided without it: every analysis and every edit names the reading (A1 / B2 / …) that
motivates it.

**When you are here.** Before any authoring (the budget and this profile come first — `budget.md`),
on the plain champion (gate input), again on the Gluon anchor after transcription, and **every
round** of the climb on each layer candidate. Profiling reads the compiled ISA and the hardware
counters, so it is independent of plain-vs-Gluon. The bound-class judgement is **yours** — this
skill ships no classifier (relative-first rule engine); the tools emit signals, you read them.

**Who profiles, in GEAK.** `profile_engineer` takes the baseline profile (Profile phase) and the
post-merge re-profile; the deep_engineer re-profiles its own intermediate versions inside its loop
(`OUTPUT_DIR/profile_rN`). Both go through GEAK's own entry, `kernel_workflow/scripts/profile_kernel.sh`,
under `gpu_lock.sh`, inside the GEAK workspace — the profiler runs where GEAK runs the kernel. There is
no "no full profile before the port lands" exemption — the anchor is profiled in full like any other
round. When the kernel under test lives in a separate container, the pack's execution-locus layer
(`scripts/locus.sh`, optional) applies (`### Execution locus`).

## Profiler entry and tool locations

GEAK infrastructure is the single owner of the profiler entry and the GPU lock. All paths below are
relative to the GEAK repo root.

| what | where | notes |
| --- | --- | --- |
| profiler entry | `kernel_workflow/scripts/profile_kernel.sh <gpu_id> "<benchmark_cmd>" <output_dir>` (+ `profile_policy.sh`) | warms up, takes `kernel_workflow/scripts/gpu_lock.sh` for every run, picks the best available profiler by arch, dumps **raw** output to `<output_dir>/profile_report.txt` (+ native artifacts). Optional modes: **`--pmc`** (memory counters + derived compute/occupancy metrics + optional SPI pass — `## Derived metrics (direct \`rocprofv3 --pmc\` read)`) and **`--att`** (thread trace — `## rocprofv3 ATT (optional \`--att\`)`), merged from the pack's former profiler |
| GPU lock | `kernel_workflow/scripts/gpu_lock.sh <gpu_id\|pool> <command...>` (lock dir `/tmp/team_gpu_locks`) | every other wrapper below runs under it: `bash kernel_workflow/scripts/gpu_lock.sh <gpu> bash kernel_workflow/scripts/kernel_tools/capture.sh …` |
| one-command evidence capture | `kernel_workflow/scripts/kernel_tools/capture.sh` | ATT rollup + static ISA audit + IR in one go (preflight → PMC-live / PMC-blind branch); per-joint scripts + degrade table: `scripts/USAGE.md` |
| safe rocprofv3 | `kernel_workflow/scripts/kernel_tools/rocprofv3_safe.sh --kernel <regex> --out <dir> [--pmc "A B C"] [--timeout 150] -- <cmd...>` | never call `rocprofv3` bare (`### Wrapper conventions`) |
| rocprof-compute per round | `kernel_workflow/scripts/kernel_tools/rocprof_compute_probe.sh <name> <out> -- python <wrapper>` → `parse_rc.py` | `### rocprof-compute per-round (aggregate SOL + memory chart + occupancy limiter)` |
| parsers / ATT | `kernel_tools/parse_pmc.py`, `parse_rc.py`, `kernel_breakdown.py`, `hotspot_analyzer.py`, `att_opclass.py`, `att_timeline.py`, `att_to_perfetto.py`, `att_merge_perfetto.py`, `tile_trace.py`, `serve_traces.py` | |
| static ISA / occupancy | `kernel_tools/dump_ir.sh`, `asm_loop_audit.py`, `asm_schedule_viz.py`, `mfma_efficiency.py`, `deep_mfma_analysis.py`, `layout_facts.py`, `probe.py`, `amd_occupancy.py`, `gfx950_isa.py` | |
| screening / harness | `kernel_tools/ab_bench.py`, `create_harness.py`, `parse_correctness.py` (timing and correctness through `e2e_workflow/scripts/harness_lib.py`) | method in `benchmark-hygiene.md` |
| budget / roofline | `kernel_tools/hw_budget.py`, `calc_perf.py`, `mem_bw_probe.py`; peaks from `perf_knowledge/hardware/data/sku.json`, cutoffs from `perf_knowledge/hardware/data/thresholds.json` | method in `budget.md` |
| pack-only | `scripts/served_envelope.py`, `scripts/env_gate.sh`, `scripts/locus.sh` (optional: kernel in a separate container) | the old `scripts/<name>` paths of moved tools are shims and still resolve |

Interpretation of the raw dump follows `kernel_workflow/knowledge/profiling_guide.md` (start at
"Reading the raw profiler dump"; on a failed profiler work its "Profiler failed? — fault-tolerance
ladder": read the error, `<tool> --help`, re-run once with the env override `RPC_PROFILE_ARGS` /
`RPV3_TRACE_ARGS` / `RPROF_ARGS` / `METRIX_ARGS` / `PROFILER_PRIORITY`, then degrade deliberately and
record it). Default priority is arch-aware: CDNA `rocprof-compute → omniperf → rocprofv3 → rocprof →
metrix → benchmark-only`; gfx1201 `rocprofv3 → rocprof → metrix → rocprof-compute → omniperf →
benchmark-only`. This chapter adds the pack's per-round reading rules on top of it; where the two
disagreed, the resolution is stated in place.

### Wrapper conventions

- Every profiler wrapper runs under `gpu_lock.sh`; **never inline `HIP_VISIBLE_DEVICES`** in a
  profiler command. Measured reason: a wrapper that exported `HIP_VISIBLE_DEVICES=0` **inside** an
  outer `ROCR_VISIBLE_DEVICES=N` made rocprofv3 abort; the campaign recorded "TCC counters hang
  rocprofv3 indefinitely" and withdrew every derived figure — with the inner variable unset, TCC
  returned in ~80 s. `rocprofv3_safe.sh` unsets `HIP_VISIBLE_DEVICES` only when **both** it and
  `ROCR_VISIBLE_DEVICES` are set (an unconditional unset mis-numbers the GPU).
- **A hard timeout** (`--timeout`, default `PMC_TIMEOUT=150` s; exit **124** = timed out, reported as
  a **degraded layer, never as a zero**). What a counter group the box cannot schedule does —
  replay, abort (SIGABRT) or **hang while holding the GPU flock** — is **version-dependent**; the
  wrapper's timeout + degrade handles all three. One campaign lost ~35 min of card time to a hang
  that looked like a slow profile. Do not apply this timeout to e2e serving captures (slow finalize
  there is normal).
- **Always filter by kernel** (`--kernel-include-regex`): it took one real run from >7 min to ~80 s,
  and unfiltered collection sweeps up torch/CK/cutlass helpers whose rows you would then average.
- Scripts accept `bash -c` command strings, write no `rm` (move stale dirs aside), and keep timing
  and profiling in **separate runs** (`benchmark-hygiene.md`).

## Profiler-capability preflight (verify ONCE — do not assume PMC is visible)

`rocprofv3 --pmc`/ATT below assumes the profiler can intercept the kernel. On most boxes it can for
Triton/Gluon, but verify rather than assume — a sibling skill found `rocprofv3` captured **0
dispatches** of a JIT-launched kernel (only torch helper kernels), silently yielding no counters:

```bash
bash kernel_workflow/scripts/gpu_lock.sh <gpu_id> \
  rocprofv3 --kernel-trace -f csv -d /tmp/kt -- python your_bench.py   # then:
grep -c <your_kernel_name> /tmp/kt/*/*kernel_trace.csv   # 0 => PMC is blind on this box
```

- **>0 dispatches** → PMC/ATT below is live; use it.
- **0 dispatches** → fall back to the **native ISA path** (`kernel_tools/dump_ir.sh` →
  `.amdgcn`/`.s` → `asm_loop_audit.py`, with VGPR/AGPR/spill from the `.amdgcn` / compile stats)
  **+ the floor probe** (`### Rule: floor probe (quantify overlap headroom exactly)`). No
  `inspect_source.py` is needed — the Triton cache already emits loose ISA. Record the live path in
  the contract's `Profiler preflight` field.

Always extract the **dispatch count** (kernels launched per call) whatever profiler ran
(`kernel_workflow/knowledge/profiling_guide.md`). `capture.sh` runs this preflight itself and records which branch it took in
`capture.json`.

**RDNA4 (gfx1201) is PMC-blind by design, and that is a degrade, not a failure.** Kernel-trace
usually records dispatches, but CDNA derived names (`SQ_WAVES`, `VALUInsts`, `MfmaUtil`, `VALUBusy`)
may be missing or mean something else; an unresolvable derived name fails the **whole** rocprofv3
job, so off gfx9 the CDNA defaults cost the collection, not a few columns. List what the build
exposes (`rocprofv3 -L` / `--list-avail`; older generations `--list-basic` / `--list-derived` /
`--list-counters`, or `rocprofv3-avail list --pmc`) and pick the equivalents; never invent MFMA
utilization (`../pitfalls/platform-known-issues.md`; `kernel_workflow/knowledge/profiling_guide.md` "RDNA4 client").

### Execution locus

**FIX before you degrade (the profiler runs WHERE the kernel runs).**

- **Default (GEAK workspace):** GEAK runs the kernel in its workspace under `gpu_lock.sh`, and
  `profile_kernel.sh` runs the profiler in that same environment — there is no separate locus to
  configure. If GEAK itself deploys the candidate into a container, the profiler must run in that
  container too (the rule below), and that is GEAK's harness configuration, not `locus.sh`.
- **Kernel in a separate container:** if the task/contract specifies a container (the harness deploys + runs the
  candidate via a `CONTAINER=`/`docker exec` path), then ALL profiler collection (`rocprof-compute
  profile --no-roof` + `analyze`, `rocprofv3 --pmc`, `rocprofv3 --att` + decode) MUST run **inside
  that same container** — a host-side profiler cannot see the in-container JIT kernel and silently
  yields `sol_pmc_no_kernel_rows` / `analyze blind` / ATT `code:null`. This is the **#1 cause of a
  "blind" profiler and it is a FIXABLE mis-configuration, NOT a blind mode.** `scripts/locus.sh` +
  `TILE_KERNEL_CONTAINER` auto-wrap the collection in `docker exec` and `docker cp` the artifacts
  out (host-side is unchanged when no container is specified). Host-side collection is valid ONLY
  when the task did NOT specify a container. `capture.sh` exits **4** (fail-closed) when a container
  is configured but the profiler cannot run in it — no artifact was written, and anything collected
  host-side would not describe the kernel under test, so the caller must not treat it as a degraded
  capture.

**Only after the profiler has been run in-locus and still fails is a degrade recorded** — a
host-side-over-a-container-kernel degrade is `profiler_locus_mismatch`, triaged as
retryable-fix-then-escalate (`triage.md ## Retryable vs scoped-ceiling vs global (continue / defer /
halt)`), never a silent terminal degrade.

#### Execution-locus path contract (separate container)

When `TILE_KERNEL_CONTAINER` identifies a live kernel container, profiling commands run inside that
container. Collection output and host-side analysis must use one working directory that is readable
and writable from both sides at the same absolute path.

- Set `TILE_KERNEL_CONTAINER_WORKDIR` to that shared absolute path before `locus_run`.
- Use `locus_timeout <seconds> <command...>` for bounded payloads. The timeout executes inside the
  kernel locus, not around the host-side container client.
- Call `locus_workdir_shared <path>` before a profile that collects in the container and analyzes on
  the host. It verifies host-to-container and container-to-host visibility.
- Treat `profiler_locus_workdir_unshared` as a mount/path configuration fault. Do not call it a
  profile or analyze failure, and do not substitute an unrelated host path.

`rocprof_compute_probe.sh` follows this contract before it starts collection. `scripts/env_gate.sh`
uses the requested workdir for its container smoke and reports unavailable locus, unshared workdir,
profile failure, and analyze failure as distinct conditions; a missing profiler or rocprof-compute
unsupported on RDNA4 is reported as a **degrade**, not a hard fail.

**Shared ownership (host ⇄ container) — why you occasionally cannot write into `exp/`.** The `exp/`
tree is written by TWO principals on the SAME bind-mounted path: the host (you edit your records /
the kernel as your login uid) and the kernel container (the profiler runs as **root** via
`locus.sh`). ROCm profilers (`rocprofv3` / `rocprof-compute`) **must run as root** — running them as
a non-root `--user` aborts (signal 6), so we cannot equalize by dropping root. Instead make your
`exp/` tree **group-writable + setgid with the bind-mount's shared gid** (`umask 0002` + `chmod
g+rwxs` + `chgrp` to the **bind-mount parent dir's gid** — nothing hardcoded, and NOT `id -g` which
is `0` inside a root container; the parent gid is the same non-zero host gid seen from either
side). setgid propagates that gid to every round/capture subdir, so a file root writes stays
host-editable and root can always write host files — and because it keys off the parent gid (not
the caller's), it works no matter which side created the tree first. If you STILL hit EACCES,
re-create the tree with the right gid (or `chmod -R g+ws` each round). Bytecode caching is off
(`PYTHONDONTWRITEBYTECODE=1`) so no root-owned `__pycache__` lands in the shared skill dir.

### Two blind modes that are NOT a failure

Degrade is a valid terminal state **after** the locus fix above. When a profiler cannot see the
kernel *even running in the kernel's own locus*, degrading to **timing + native ISA** (or ATT-only)
is a **legitimate, complete** evidence path — record it in `caveats[]`, do not treat it as an
incomplete round:

- **CUDA-graph + external JIT (`.so`) → rocprofv3 kernel-trace is blind.** When the op runs the
  kernel inside a **CUDA/HIP graph** and/or launches it from an **externally JIT-compiled `.so`** (a
  `dlopen`'d code object), the dispatch is hidden inside the graph replay and rocprofv3
  `--kernel-trace` sees 0 (or only the graph-launch) — even though the kernel really runs. Route to
  **timing (graph-boundary A/B) + native ISA** (`dump_ir.sh` / `llvm-objdump` the loaded
  `.co`/`.hsaco` → `asm_loop_audit.py`) + the floor probe. This is the primary evidence path for
  graph-served production ops.
- **`rocprof-compute` fails (rc=4 / "no workload dir").** On some boxes `rocprof-compute` /
  `rocprof_compute_probe.sh` exits non-zero (rc=4) or leaves no workload dir. That is not a stop —
  **degrade to `rocprofv3 --pmc` / ATT, or to timing+ISA** and name the degrade in `caveats[]`. A
  JSON config passed via `docker exec bash -lc '…'` argv can itself trigger rc=4 /
  `JSONDecodeError`; pass the config by an **env-var path**, not argv (`benchmark-hygiene.md ##
  Profiler output hygiene (do not pollute the repo)`).

## 3.1 Required evidence — the four dials, every round

**Every decision this skill makes is profile-guided. This table is not a menu — it is the required
instrument set, and the four groups A/B/C/D must all be read.** Without them you are guessing at a
black box: you cannot see the register wall (A), what the instruction stream is actually made of (B),
whether you are bandwidth-bound (C), or whether a change was real (D). The round-loop discipline
that consumes these readings (gather before hypothesising, name the reading behind every edit,
report a missing number as missing — never as a zero) is in `climb.md ## 3. The round loop`.

**A — occupancy and registers**

| | what you must know | tool | bad-value tell |
| --- | --- | --- | --- |
| A1 | `next_free_vgpr` / `agpr` / `spill` → **waves/SIMD**, then **converted to `wg/CU`** | `asm_loop_audit.py` (auto via `capture.sh`) | 1 wave/SIMD; any hot-loop `scratch_*` is real spill. **Two ways this row lies.** (a) It reports **waves/SIMD**; the residency that feeds bandwidth is **`wg/CU`**, and a workgroup is indivisible, so convert and carry `num_warps` as a column — they coincide only at `num_warps == simd_per_cu`, and one measured 240→146 VGPR cut bought **zero**. (b) LLVM's `; Occupancy: N` is authoritative for the **register term only** — no LDS term, so on the dynamic-LDS kernels Gluon emits it can overstate by **3x** and the hand `min(reg, LDS)` is *more* correct. Both derivations, with the conversion formula and the thresholds that still move it: `../hardware/planning-constants.md`; the arbiter is `kernel_tools/amd_occupancy.py` |
| A2 | per-tensor register ownership **derived from the tile plan, pre-compile** | `probe.py` | persistent set > 256 ⇒ this tile can never reach 2 waves/SIMD. The *pair* gate's VGPR half is vacuous at `num_warps >= 4`, where its ceiling is the architectural per-thread max: a config can read `vgpr_count: 256`, "pass", and carry **345 spill slots** — demand changed address space, it did not shrink. Read `.vgpr_spill_count` from the same row (`../hardware/planning-constants.md`) |
| A3 | **LDS bytes/WG → WGs/CU** | `asm_loop_audit.py --meta <ir-dir>` | ⚠ source is the Triton cache metadata `shared`. The KD's `group_segment_fixed_size` and rocprof-compute `7.1.8` are **structurally 0 for every Triton kernel** — a 0 from those is not evidence |

**B — what the instruction stream is made of**

| | what you must know | tool | bad-value tell |
| --- | --- | --- | --- |
| B1 | **ranked inter-MFMA bubble ownership in measured cycles**, plus the cadence verdict (issue-bound vs bubble-diluted) | `mfma_efficiency.py` (ATT, auto via `capture.sh`) | a lever that "worked" without moving its own bucket did something else — re-read this after every change |
| B2 | the feed bucket **resolved: LDS (lgkmcnt) vs global (vmcnt)** | same rollup | the two halves are one bucket over two resources with **opposite** fixes (layout/swizzle vs coalescing/async copy) |
| B3 | **per-opcode hot-loop histogram + diff vs last round** | `asm_loop_audit.py --opcodes` | op-class ("valu 64.5%") is one level too coarse: a dtype-conversion sequence, an address chain and a layout permute all land there. Keep `s_waitcnt` in the diff — it moves when only the *schedule* moved (D3) — and read the whole histogram, not the mnemonic you came to count |
| B4 | MFMA operand-layout facts **before** picking a tile | `layout_facts.py` | two shapes with equal MAC/cycle can differ 4× in accumulator GPR/lane |

**C — the memory side (the one bound you cannot infer from the ISA)**

Every "% of roofline" or "gap" read off these rows carries `numerator_basis` (model | counters) and
`denominator_basis` (datasheet | empirical@tool-version | in-shape probe); a datasheet denominator
ranks only, and only a measured numerator over a probed/calibrated denominator may gate or close
(`budget.md`, and its checklist "3.3 Before you report a "% of roofline" or a "gap = X us"").

| | what you must know | tool | bad-value tell |
| --- | --- | --- | --- |
| C0 | **does a roofline apply at all** | `hw_budget.py --grid <wgs> --footprint-mb <unique> --dispatches <n>` | check FIRST: nothing in C1–C4 means anything when the model is inapplicable. Fewer workgroups than CUs ⇒ part of the machine is idle by construction and **neither** roofline term applies (tell: measured time barely moves while bytes change by an order of magnitude; fix is parallelism, not locality). A working set inside the **memory-side LLC** ⇒ the **memory** floor is void in the steady state of a repeated-iteration benchmark, though the compute floor survives and is then binding (tell: measured time *below* the memory floor). Several dispatches inside the timed region ⇒ a per-kernel floor cannot be compared against the region total. Also: filling every CU **once** still cannot saturate memory — bandwidth is `requests-in-flight × bytes-per-request / latency`, so quote the ceiling probed at *this* program count |
| C1 | achieved **fabric** bandwidth + L2 hit rate | `rocprofv3 --pmc FETCH_SIZE WRITE_SIZE TCC_HIT_sum TCC_MISS_sum` (through `rocprofv3_safe.sh`) → `parse_pmc.py` (memory block: `TCC_MISS*128 B / dispatch duration`, plus the independent `FETCH_SIZE+WRITE_SIZE` and `TCC_EA0_*` routes) | without it you only *assume* you are not bandwidth-bound, and everything downstream inherits that. Read it against the **in-shape ceiling (C3), never the datasheet peak**. Two things this number is **not**: (a) it is not DRAM traffic — every one of these counters is sampled at the L2/fabric boundary, in front of the memory-side LLC, so it cannot see whether a byte came from that cache or from HBM (that is C0's job); (b) it is not a single number — the routes agree to a fraction of a percent on a pure stream and can diverge **several-fold** on a real kernel, because whole-line miss counting is blind to partial-line stores, write-allocate refills and atomic round-trips and therefore reads **low**, the direction that fakes *memory does not bind*. A divergence that reproduces on a known-bytes probe is a unit slip; one that appears only on the kernel means the routes measure different things. Pass them all and carry the width. **gfx950: `FETCH_SIZE` / `TCC_BUBBLE` under-count read bytes** — another reason never to read one route alone. **≤4 derived counters is a STARTING budget, not a guarantee** — the slot limit is box- and version-dependent (some gfx942 boxes SIGABRT even at 4; others replay or hang — `### Wrapper conventions`), so on an abort/timeout BISECT the list (`collect_pmc_group()` in `parse_pmc.py`) and drop only the counters that overflow alone, instead of losing the whole metric |
| C2 | intensity vs ridge; the binding floor and your multiple over it | `hw_budget.py` (+ `--measured-dram-mb` / `--measured-hbm-tb-s` once C1/C3 exist) | uncalibrated it divides an analytic byte model by a datasheet peak — both optimistic, and they compound. It prints `[datasheet]`/`[model]` vs `[calibrated]` for exactly this reason, and **refuses** `%-of-ceiling` and the gap until the denominator is a probed one. The caveat is scoped to the axis that BINDS, which is what makes it readable: a compute-bound kernel is sent to probe its MFMA ceiling and not `mem_bw_probe.py`, and until it does, its multiple over the floor is reported as an **upper bound** on the prize rather than as the prize. Off-axis warnings are how on-axis ones stop being read. A measured time **below** the floor is not a record, it is the model announcing it does not apply (the tool exits non-zero). Do not stretch a named archetype (`--workload gemm\|attention\|moe`) onto a kernel that only matches it in name: one `dtype` cannot carry per-tensor precisions (a low-precision-operand GEMM with a wide output has most of its bytes in the **output**), an argument list charges pointers the kernel never dereferences, allocation is not traffic, a tile loop's re-read multiplier comes from the **tile** and not the shape, a dead grid dimension duplicates every program, and `q_len==1` is a gather rather than the quadratic thing its name implies. Declare what the kernel **moves** instead: `--tensors "name:dir:dtype:dims[:xN]" --flops <n>`, which returns a bracket (unique footprint → issued traffic) rather than a false point. And a FLOP count is not yet a compute floor: `flops / MFMA_peak` is one only if the FLOPs go through **MFMA**, so declare the engine (`--flops-engine mfma\|valu`; a named `--workload` carries it). The `c*n` archetypes — reduction, elementwise, norm, scan, gather — have no matrix multiply at all, and pricing them at the matrix rate puts the compute floor one to two **orders** too low, after which it never binds and `memory` falls out as a *default rather than a finding* — wrong in exactly the direction that opens a byte-removal round on a kernel whose arithmetic is the constraint. Likewise a **missing** ceiling is not a zero ceiling: an absent per-dtype rate (CDNA4 fp4/fp6, int8 on an Instinct row, a typo) is reported, never substituted with bf16's — too low a peak *lifts* the compute floor and can flip the binding onto an engine that was never the constraint |
| C3 | the **in-shape** ceiling on the axis that binds — memory: this run length, stride, read:write mix and program count; compute: this dtype | `mem_bw_probe.py --sku <sku> --runlen .. --stride .. --rw-mix ..`, and for the compute axis a dense back-to-back MFMA loop → `hw_budget.py --measured-tflops <lo>,<hi>` | the datasheet peak is unreachable at any real access shape, so a gap measured against it can be a multiple of the real one — enough to say "keep going" where the truth is "close it out". Probe on **this** box: even the read-vs-write asymmetry is per-part. A ceiling is a **range** (a few percent of session drift, and it rises with parallelism): quote `lo-hi` with the program count, and treat a gap narrower than the range as inside the error bar. **The compute peak is no safer than the memory peak**: the datasheet MFMA rate assumes back-to-back issue with operands already in register, which a kernel that also loads, converts and addresses does not sustain — so a compute-bound kernel needs this row just as much, and calibrating it moves the multiple over the floor DOWN (a datasheet floor sits too low, so it can only flatter the prize) |
| C4 | does memory **bind at all** | floor probe (`### Rule: floor probe (quantify overlap headroom exactly)`) | REQUIRED before you call a kernel memory-bound and spend a round removing bytes. If `non_stream_floor > memory_ideal`, the memory system is not the constraint and the roofline gap is not a prize: making both streams cache-resident can leave a kernel *slower* than its memory ideal, which prices the byte-removal lever at the overfetch alone rather than at the whole gap |

**D — timing and correctness**

| | what you must know | tool | bad-value tell |
| --- | --- | --- | --- |
| D1 | a timing you can compare: **acceptance** numbers from GEAK's harness (`e2e_workflow/scripts/harness_lib.py`: CUDA events with per-sample sync, read-evict flush, **median**, fresh process per leg, same-window baseline; commit gate `MIN_IMPROVE=2%`); **search/screening** numbers from a same-window **interleaved** A/B with median **and** spread (min as a supplementary column) | acceptance: harness_lib (`verify_engineer` / `measure_legs`); screening: `scripts/ab_bench.py` (control arm, `--permute`) | spread wide enough to contain your claimed delta. Two numbers from two different windows are not comparable on a clock-unstable box — an acceptance leg is a fresh process, but its baseline is measured in the same window. A measured noise band may only make a verdict **stricter** than 2%, never looser (`benchmark-hygiene.md`) |
| D2 | correctness oracle **in-process, gating before timing** | your harness; the champion's own oracle command | any timing recorded for a variant whose oracle did not run |
| D3 | after adding a default-off `constexpr` knob: the shipped kernel's **filtered instruction stream still hashes to the pre-knob value** | strip directives / labels / comments, join, hash; compare against the same file with the branch **deleted**, not set to 0 | a knob is free by semantics, not by schedule: one spelling added **6 `s_waitcnt`** to the shipped config with A1, B3 and the oracle all passing, after which the timing table no longer described the kernel (`benchmark-hygiene.md ### A knob that is off still has to prove the shipped stream did not move`) |

### Required evidence (record every round)

The per-round record that the four dials fill (field names as written into the per-round profile
JSON; `round_<n>/record.json`, `## Output`):

```text
profiler_used: rocprof-compute | omniperf | rocprofv3 | rocprof | metrix | benchmark_only
per-shape latency + TFLOPS = 2*M*N*K/(t_us*1e6)
dispatch count (kernels launched per call)
MFMA efficiency = mfma_cycles_in_loop / avg_iter_duration ; phase ratios (pro/loop/epi)
clock-insensitive A/B discriminator (take at the anchor): MfmaUtil, VALUBusy,
  MemUnitStalled, dependency-wait / issue-wait (each over Total wave cycles)
  -- preferred over absolute time on a drifting box
achieved HBM BW = bytes/cycles ; arithmetic intensity ; bound class + binding sub-resource
  (every % with numerator_basis + denominator_basis)
ds_read_b128 steady interval (16 = conflict-free; 32/64 = 2/4-way)
occupancy: resident WG, active waves ; VGPR / AGPR / spill counts (from the static AMDGCN dump)
rocprof counters: TCC_EA0_RDREQ_DRAM_sum (L2->DRAM), TCP_TCC_READ_REQ_sum (L1->L2)
toolchain/env: ROCm; upstream Triton minor; fork lineage + tag; LLVM revision; host build
  (exports symbols? keeps TargetMachine for plugins?); any pass plugin + the revision it was
  built against; the compile options ACTUALLY set (llvm_fn_attrs, num_warps, waves_per_eu, ...)
  rather than the ones intended; TRITON_HIP_USE_COEXEC_SCHEDULER (3.8.0 default on);
  TRITON_CACHE_DIR; rocprof-compute / rocprofv3 version
  # a number whose toolchain identity is unknown cannot be compared to one taken elsewhere
  # -> ../tile-programming/compiler-contract.md ## Toolchain identity
```

## 3.2 Evidence layers: the floor, and the full read

**The floor — both work with `rocprofv3` alone:** ATT warp trace (B1, B2) and static ISA (A1, A3,
B3). A round can never fall below this pair.

**Collect `rocprof-compute` SOL when the profile trigger requires it.** Its non-redundant signal is
larger than "just the memory block": the **memory block** (achieved HBM/MALL BW + L2 hit — the one
bound you cannot infer from the ISA, C1), **`warp_state`** (dependency-wait / issue-wait cycles =
the AMD analog of Nsight stall analysis), and **coalescing** — none of which the ATT/static floor
gives you. Its occupancy block *is* redundant with A1 (free), but that is one block, not the reason
to skip it.

**A SOL failure on a PMC-capable box is a wiring bug to FIX, not a degrade to accept.** The two
historical failure modes are now guarded in `rocprof_compute_probe.sh`: the container-locus path
(fix the locus — `### Execution locus`, and the shared-workdir path contract it depends on,
`#### Execution-locus path contract (separate container)`) and a raw JSON config on the app argv
(rocprof-compute strips the quotes on replay — pass a config-embedded wrapper). Fix the wiring and
you get the full read for free. Only a genuinely **PMC-blind box** (JIT-invisible kernel, no
`rocprof-compute` installed, or RDNA4's missing CDNA counters) is a legitimate terminal degrade —
note it in `caveats[]` and proceed on the ATT + static floor.

## Report-reading heuristics

When a rocprof-compute / omniperf report is available, focus on high-signal sections (gfx950 is the
main line; recalibrate per part — gfx942 downgrade: 304 CU, HBM 5.3 TB/s MI300X / 6.0 TB/s MI325X
vs 256 CU, 8.0 TB/s on MI355X/MI350X, `perf_knowledge/hardware/data/sku.json`):

| Section | Read |
| --- | --- |
| System Speed-of-Light | "Pct of Peak": MFMA / VALU **busy**, VMEM, LDS util; HBM BW — vs the **probed in-shape** ceiling (C3); the datasheet ~8 TB/s (gfx950) only ranks |
| Wavefront Runtime Stats | Active, Dependency Wait, Issue Wait — each **divided by Total Wave Cycles** (independent accumulators over the same total; never subtract one from another — that yields negative "active", a mis-read). Dependency wait = **latency C1** (serial dependency chain), issue wait = **latency C2** (too few resident waves) — neither is "memory" vs "compute" |
| Compute Pipeline | MFMA / VALU busy and throughput; VALU active threads vs this card's wavefront (wave64 on CDNA: <64 ⇒ divergence) |
| Cache (L1/L2) | L1 hit (>60% non-streaming), L2 hit (<50% => streaming/random) |

Precise rocprof-compute **block IDs** for each node above (SOL `2.1.x`, SPI occupancy limiters
`6.2.x`, roofline `4.x`, launch `7.1.x`) are tabulated in `../hardware/bound-class-signals.md ##
rocprof-compute block-ID cheat-sheet (when a rocprof-compute report is available)` (probe per build;
do not hardcode). Version caveats: `--roof-only` on gfx95x needs rocprof-compute **>= 3.6.0**;
gfx950 `FETCH_SIZE` / `TCC_BUBBLE` read bytes under-count (C1).

Diagnostic ratios + severity (single source for the tree + cutoffs:
`../hardware/bound-class-signals.md ## Bound-class decision tree (exhaustive, ordered, defaulted)` +
`perf_knowledge/hardware/data/thresholds.json`, implemented by your own reading of the evidence; the
interpretation follows `kernel_workflow/knowledge/profiling_guide.md` "Section 7.2: Wavefront Runtime
Stats" and "Bottleneck Classification Decision Tree" — the ratios below are a reading aid, not an
authority):

```text
kernel_efficiency = Active / Total_wave_cycles
   <20% CRITICAL ; 20-60% MODERATE ; >60% ACCEPTABLE
Dependency Wait / Total dominant  => latency-bound C1 (serial dependency chain feeding the
                                      math units) -> shorten the chain.  NOT memory-bound.
Issue Wait / Total dominant       => latency-bound C2 (too few resident waves to issue from)
                                      -> raise occupancy / fill the GPU
MfmaUtil or VALUBusy (busy counters) >60% of peak => compute-bound (sub-resource = the busy one)
memory-bound ONLY when HBM/VMEM busy is actually high (>=60% SoL) or L1 locality is the
   demonstrated problem, AND the C4 floor probe says memory binds -- low arithmetic intensity
   alone does not make a kernel memory-bound
very short kernel + low util everywhere => latency / launch-bound
elevated LDS bank-conflict metrics => lds-bound
all metrics 20-60% => balanced (algorithmic/fusion opportunity)
```

- C1 and C2 have **opposite** fixes (C1 often wants more registers per wave for a shorter chain; C2
  wants fewer registers for more waves). The split is a property of the **configuration**, not the
  source: small tiles push a serial preamble (dequant, scale, address math) to dominate → C1; large
  tiles grow the accumulator → fewer waves → C2. One measured kernel read **72% dependency wait** at
  the small tile and **42% issue wait** at the large one — same source, opposite fix. Re-read the
  split after every change to tile / `num_warps` (and `num_stages` on the plain side — it is dead on
  the Gluon path) instead of carrying the prior diagnosis forward.
- These are **averages over every wave in the dispatch**; a kernel whose waves diverge (MoE experts
  with uneven token counts, ragged attention) can average into a split that describes none of its
  waves — re-profile a single representative launch when the split is ambiguous.
- Before trusting a label, run the cheap raw-counter checks in `kernel_workflow/knowledge/profiling_guide.md` "Cheap checks to
  run from the raw counters BEFORE trusting a label" (an efficiency >100% is a mis-calibrated peak,
  HBM under-report on multi-XCD, MoE padding inflating AI, `CTAs < CU count` fill, spill first, LDS
  bank-conflict and coalescing ratios).

Ignore metrics showing 0/nan/empty (not active for this kernel).

Bound-class discrimination: `../hardware/bound-class-signals.md` — enter through its **`## Bound-class
decision tree (exhaustive, ordered, defaulted)`** (exhaustive, busy-not-util, latency/occupancy as
the forced residual), then refine on the discriminator leaves; on the latency residual read **`##
Stall-reason → lever (latency/occupancy sub-classification)`** + the SPI occupancy limiter. The
`s_waitcnt`/`s_nop` readings from `asm_loop_audit.py` map to that same stall-reason table (usable in
degraded/no-PMC mode).

## Derived metrics (direct `rocprofv3 --pmc` read)

When no rocprof-compute / omniperf report is available, read the binding compute sub-resource and
occupancy **directly** with `rocprofv3 --pmc` (the `--pmc` mode of `profile_kernel.sh` runs this as
one extra pass alongside the memory counters; or `rocprofv3_safe.sh --pmc "…"`):

```text
MfmaUtil          matrix-core BUSY/throughput (fraction of ALL cycles MFMA worked)
VALUBusy          vector-ALU BUSY/THROUGHPUT (fraction of ALL cycles VALU worked)
                  -- the VALU throughput-bound discriminator (read THIS, not Utilization)
VALUUtilization   lane-occupancy DURING active VALU cycles (NOT throughput; can read
                  ~100% on a kernel that is not VALU-bound -- see the rule below)
LdsUtil           LDS-unit busy
LDSBankConflict   % of LDS accesses that bank-conflict
OccupancyPercent  achieved occupancy (fraction of max resident waves)
MemUnitStalled    % of cycles the memory unit stalled (~0 => not memory-bound)
```

Default gfx950 sets of the former pack profiler: memory counters
`TCC_EA0_RDREQ_DRAM_sum,TCP_TCC_READ_REQ_sum`; derived
`MfmaUtil,VALUBusy,VALUUtilization,LdsUtil,LDSBankConflict,OccupancyPercent,MemUnitStalled`. These
derived names are **CDNA (gfx9) only** — off gfx9 they are skipped and you list the build's own
counters (`## Profiler-capability preflight`). **Occupancy-limiter attribution:** the measured SPI
(Workgroup Manager) "Insufficient CU" stats are the ground-truth cross-check for the analytic
`min(by_LDS, by_regs, by_waves)` (`../hardware/bound-class-signals.md ## Bound-class decision tree
(exhaustive, ordered, defaulted)`, `../hardware/roofline-models.md`). Cleanest via rocprof-compute
blocks `6.2` + `2.1.15` (6.2.5 VGPR / 6.2.7 LDS); raw rocprofv3 SPI hardware-counter names are
**build-specific** — probe them per box rather than hardcoding.

### Rule: a single utilization at 100% is necessary-not-sufficient

A resource reading ~100% does **not** prove the kernel is bound by it — a unit can read ~100% while
it is actually **stalling** on another resource (it keeps issuing into the wait). The same
`VALUUtilization=100%` can sit on a genuinely VALU-bound kernel *and* on an LDS-bound one.
Discriminate before choosing a lever:

- **cross-check** the other derived metrics (`MfmaUtil` / `LdsUtil` / `LDSBankConflict`) and the
  roofline class;
- run a **controlled A/B**: remove some of that resource's work (e.g. drop one VALU op) and check the
  time actually moves. If removing the work does not speed it up, the kernel is bound by something
  else and the 100% was a stall artifact.

### Rule: read the busy/throughput counter, not the duty-cycle one

Vendor counters expose **two different things with confusingly similar names**, and a
`*Utilization` reading is the *wrong* one for classifying a compute bound:

- a **duty-cycle / lane-occupancy** counter (`*Utilization`): "of the cycles this unit *did* issue,
  how full were the lanes" — it can sit at ~100% on a kernel that is **not** bound by that unit,
  because it ignores the cycles the unit *could* have issued but did not;
- a **busy / throughput** counter (`*Busy`): "what fraction of *all* cycles this unit was actually
  working" — this is the one that bounds the kernel.

When classifying a compute sub-resource read the **busy/throughput** counter (on gfx950: `VALUBusy`,
`MfmaUtil`), **not** the duty-cycle one (`VALUUtilization`). The SoL thresholds in
`kernel_workflow/knowledge/profiling_guide.md` ("VALU > 60% = compute-bound") are read on the busy counter. Discriminant: if
`VALUUtilization` ~100% but `VALUBusy` **and** `MfmaUtil` are **both** well under 100% (both compute
units half-idle) **and** `MemUnitStalled` ~0, the kernel is **dependency/latency-bound (C1) at its
current occupancy**, not throughput-bound — the units idle waiting on a dependency chain (+ LDS-read
`lgkmcnt` latency), not on a saturated pipe. Mislabeling this as "at the ceiling" sends you to
algorithmic VALU-reduction (a capped lever) instead of the real lever (shorten the critical path /
raise occupancy). Confirm with the floor probe below before concluding "at the ceiling."

**Corollary — cutting an idle unit's instruction count is neutral.** (Indexed in the gating-law SoT,
`../hardware/bound-class-signals.md ## Lever gating laws (single source of truth)`.) When a unit's
`*Busy` is well under 100% it is **not** the binding resource, so reducing *its* instructions (fewer
`convert_layout` / `v_accvgpr_read` / a different MFMA shape / op-count micro-opts) only shortens the
slack of an already-idle unit — it does not touch the binding dependency-chain latency, so expect
**neutral**. On a latency/dependency-bound kernel the only productive levers are **shortening the
critical path** (fold critical-path scalars off the chain — `### Reducing compute-class VALU (fold
scalars off the tile)`) or **raising occupancy**; reduce-op-count and instruction-overlap are not.

### Rule: floor probe (quantify overlap headroom exactly)

Generalize the A/B above into a **floor probe**: delete the *entire* dependent stage (e.g. the whole
softmax body, leaving only the matmuls + memory), keep it compiling and running (correctness
intentionally void — timing only), and measure.

```text
exposed_fraction   ~= (full_time - floor_time) / full_time
overlapped_fraction ~= 1 - exposed_fraction      (of the deleted stage's cost)
```

The floor is the best any scheduling/overlap could reach; `full - floor` is the serial residue that
only a **structural** change (pipeline / different parallelization) can attack. Run the floor probe
**before** investing in pipelining/overlap: if the floor is itself far below peak, the dependent
stage is not even the wall — stop chasing its overlap. The same probe is C4 (does memory bind at
all).

**Stage decomposition (chain-attribution variant; time-share only).** The same mechanism, run in the
other direction, attributes time across a FUSED multi-stage chain (e.g. gather / dequant / dot1 /
softmax / dot2): gate stages cumulatively with a constexpr (`PROBE>=1, >=2, ...`) and read the
per-stage deltas. "floor probe" is the canonical name; the two directions (top-down delete vs
bottom-up cumulative-add) are the two halves of the same probe ("hourglass" is only an informal
label for the pair, not a new term). Measure kernel-only under a CUDA graph.

Be honest about what it does and does NOT give: `rocprof` counters, achieved-roofline, and budget
calibration are WHOLE-kernel — one fused launch cannot split counters per stage. So the primary
output is the per-stage **time SHARE** (which stage to attack first), NOT a per-stage bound
measurement. To classify a single stage's bound, INFER it from: (i) the per-stage
**feeds-and-speeds** budget (`../hardware/roofline-models.md` — its `T_mfma`/`T_smem`/`T_exp` terms
are already per-stage and analytic; calibrate the peak-rate constants whole-kernel/microbench, then
apply per-stage); (ii) the stage's structural nature (a scattered-load stage -> memory/latency, a
matmul stage -> MFMA, a softmax stage -> exp/VALU); (iii) OPTIONAL approximate per-stage counters by
`rocprof`-ing the GATED single-stage variant (or differencing cumulative variants) — valid but
approximate, because the gated variant re-schedules and its occupancy may differ. So a stage's bound
is INFERRED (theory + structure + time-share, optionally confirmed by gated-variant counters); the
whole-kernel counters/roofline classify only the AGGREGATE bound.

Applicability plain vs Gluon: mechanically usable on both, and a good COARSE locator on plain; but on
plain the per-stage deltas are only approximate because the automatic pipeliner/scheduler
re-schedules the reduced kernel (deltas may not sum), multi-wave occupancy hides part of a deleted
stage's cost, and gating can DCE a stage's loads/registers and shift the probe's occupancy. Trust
only the dominant-stage signal on plain; the CLEAN per-stage attribution comes after transcription
on the explicit (fixed-schedule, no-auto-pipeline) Gluon anchor. Flow placement: a Stage-Plain
profiling sub-step to localize the dominant stage on a fused kernel; its output (a stage priority)
feeds — does not replace — the bound classification below, and the same probe is reused after
transcription in transcribe -> re-profile -> re-calibrate (`transcribe.md`, `budget.md`).

**A share measured by capturing the component in isolation is an INTERVAL, not a number.** Isolating
a component removes the overlap it had with its neighbours — separately captured kernels cannot
overlap, while consecutive kernels inside the full graph do — so isolated shares are systematically
INFLATED, and the signature is that the parts sum past the whole (in one measured instance, by
**15-28%** in every bucket). Two consequences:

- **Report the share as a range**: `isolated / whole` at the top end, `isolated / sum-of-parts` at
  the bottom. Collapsing it to one number is an error, **including inside a "drive this component to
  zero" counterfactual** — which is exactly where the collapse survives review, because the
  counterfactual reads as arithmetic rather than as a measurement.
- **Amdahl on raw isolated shares OVER-predicts the op-level return** of a stage speedup, because
  part of the accelerated stage's time was already hidden under a neighbour. Pre-registering a target
  magnitude computed that way sets a number the edit cannot reach, and a real win then fails its own
  acceptance criterion.

Keep this distinct from the two instrument faults it resembles: the absolute-value invariant that
forbids a PART exceeding the WHOLE still holds (`benchmark-hygiene.md ## Every within-comparison
stays green when both arms run the wrong thing`), and a per-sample fixed cost charged once per split
is a different inflation (`benchmark-hygiene.md ## Hot or cold cache: pick it from the bound class,
because it can REVERSE a ranking`) — **which the sign of the implied fixed overhead does not separate
from this one**; what separates them is that the fixed-cost inflation scales with the number of
splits. Overlap loss is a bias of the isolation METHOD, bounded and reportable; those two are broken
measurements.

### Rule: decompose a saturated VALU before picking a lever

> Lever single source: `../hardware/bound-class-signals.md ## Lever cards (bound-class ×
> applicability)` (+ `../hardware/lever-cards.json`, filtered by your measured bound; query with
> `scripts/lever_index.py`). The recipes in this and the next two subsections are the how-to detail
> behind those cards, not a second lever registry.

"VALU-bound" is not one thing. When **`VALUBusy`** is the binding metric (genuinely
VALU-throughput-bound — confirmed by the busy-counter rule above, not a `VALUUtilization` ~100% stall
artifact), histogram the loop's VALU ops (from `.amdgcn`,
`../tile-programming/compiler-contract.md ## IR dump workflow`) into four functional classes, because
the lever differs:

| VALU class | example ops | lever |
| --- | --- | --- |
| compute | `v_exp` / `v_mul` / `v_add` / `v_fma` | reduce/fuse the math (fold scalars; `## Bound classification -> primary metric` VALU row) |
| layout-convert | `v_perm` / `permlane*` / `v_cndmask` / `ds_bpermute` | a data-layout/source change (no asm reorder removes it; `../hardware/planning-constants.md ## Extended planning (attention / fused kernels)`) |
| register-shuffle | `v_accvgpr_read` | AGPR<->VGPR form; gate by bound class (`../tile-programming/compiler-contract.md` RA-hint risk) |
| integer / address (IOPs) | index math, `v_add_u32`/`v_lshl*`/`v_mad_u*` on offsets, scalar address arith | cut address arithmetic: hoist/coalesce index math, precompute per-tile offsets, widen loads. Recognize when SOL `2.1.1 VALU IOPs` ≫ `2.1.0 VALU FLOPs` (the "instruction-bound" case; e.g. a hot loop dominated by index/offset integer ops) |

The dominant class names the real bottleneck; on result-reuse / softmax-class kernels the convert +
register-shuffle classes often dominate over the actual math (hardware reasons:
`../hardware/planning-constants.md ## Extended planning (attention / fused kernels)`).

### Reducing compute-class VALU (fold scalars off the tile)

When the *compute* class dominates a genuinely VALU-bound kernel (confirmed by the A/B test above),
the lever is to do **less per-element VALU**, not to overlap MFMA. The general move is to **fold
per-element scalar work off the hot tile**.

**General principle — fold onto the cheapest carrier on the dependency path.** A broadcast scalar
applied to every element of a tile should be folded onto the **cheapest carrier on its dependency
path — ideally an operand loaded/produced once upstream, not the per-output-tile result.** Enumerate
the carriers (input operand -> accumulator init -> per-row vector -> per-element) and fold onto the
earliest/cheapest one; verify in asm the per-element op is gone (the compiler may already fuse some
forms — treat as neutral until the asm confirms removal).

- **Fold a per-element scalar multiply into a precomputed scalar / per-row vector.** A constant or
  per-row factor applied to every tile element is cheaper folded once than applied per element. Math
  fact enabling the softmax case: `exp(x) = exp2(x * log2e)`, so `exp(qk*s - lse)` ->
  `exp2(qk*(s*log2e) - lse*log2e)` moves the per-element `*log2e` onto the per-row `lse` (and the
  compiler fuses `qk*scale2 - lse2` into one FMA + `exp2`). For a `Q@K^T * s` pattern the same `s`
  can instead be folded into **`Q` at load** (one cast over the small Q tile, hoisted out of the
  K-loop entirely), removing the per-block result scaling **and** lowering result-tile register
  pressure (`../workloads/attention.md ## Online softmax`).
- **Fold a loop-invariant additive bias into the MFMA accumulator init.** A per-row term subtracted
  from a matmul result (`mma_result - bias`) can seed the MFMA accumulator with `-bias` instead of a
  per-element subtract — but verify in asm it actually removes the subtract (the compiler may already
  fuse it; treat as neutral until proven).

Gate both by the A/B test (they are neutral / negative on a kernel that is not actually
compute-VALU-bound — e.g. an LDS-bound kernel where the extra register/schedule pressure regresses).

### Raising exp throughput by partial FMA emulation (when the exp unit is the bound)

A different case: the binding sub-resource is the **transcendental (exp) unit itself** — its busy
counter saturates while MFMA/VALU sit idle (`exp` runs on the quarter-rate transcendental path,
`../hardware/planning-constants.md`). You cannot fold the exps away (softmax needs them), but you can
**raise aggregate exp throughput by splitting the work across two units**: compute a *fraction* of
the `2^x` evaluations with a polynomial on the FMA/VALU pipe while the rest stay on the hardware
transcendental unit, so both run concurrently.

- **Mechanism**: range-reduce `x` (Cody-Waite: `2^x = 2^floor(x) * 2^frac` — the integer part by
  bit-manipulating the IEEE-754 exponent field, the fractional part by a Horner-FMA polynomial), then
  recombine. A **degree-3** polynomial matches the hardware `exp` to within ~1 BF16 ULP — sufficient
  when the softmax output is consumed in BF16.
- **Partial only**: emulation costs extra registers (coefficients + intermediates) and has longer
  latency, so emulating *all* exps risks spills that negate the gain. Emulate a **fraction** (tune by
  the measured MMA:exp / VALU:exp throughput ratio) and leave the rest on the hardware unit. Guard
  with the **tri-lemma** (`../tile-programming/slicing.md ## Occupancy budget (P8)`): the extra
  registers must not drop waves/CU.
- **Gate hard**: pays only when the exp unit is the *proven* bound. If the kernel is latency /
  `ds_read`-bound (busy counters low) the exps are already hidden and emulation is
  neutral-to-negative. The asymmetric-scaling trend (`../hardware/roofline-models.md ## Asymmetric
  scaling`) makes the exp unit more likely to bind on faster-MMA arch (gfx1250) than on gfx950.

### Rule: low MfmaUtil has two distinct tile-size causes — read the geometry to tell which

A low `MfmaUtil` on a matmul / attention loop is a **signature**, not a fix. Before touching the tile,
distinguish two causes with the same symptom but opposite levers
(`../hardware/bound-class-signals.md ## under-fill vs occupancy (the low-MfmaUtil fork)`):

- **under-FILLED (geometry).** The warp partition leaves a warp fewer than one matrix-instr tile
  along its axis (e.g. fewer rows than the MFMA M extent) → the unit issues half-empty ops. Recognize
  from the warp/instr geometry: `tile_dim < warps_on_axis × instr_extent`. Lever: **grow** the tile
  to fill it (worth a register spill — a half-empty matrix op wastes more than the spill), or do
  **not** partition a small axis (put all warps on another axis). Arch-agnostic — holds for any
  fixed-tile matrix engine (MFMA / WMMA / mma.sync).
- **under-AMORTIZED (work ratio).** The tile **is** filled, but a serial per-row stage between
  matmuls (softmax / reduction / rescale) is large relative to the matmul work per tile → MFMA idles
  behind the serial VALU. Recognize from `VALUBusy` being the bound + a fixed serial stage per block.
  Lever: a **bigger tile along the parallel (M) axis** amortizes the fixed per-row cost over more
  matmul; the plain-Triton autotuned tile is a free oracle (if plain also picked the small tile, a
  bigger one likely won't help). Gate on the occupancy budget (`../tile-programming/slicing.md`).

Both read "tile too small," and they can point opposite ways — fill needs a *bigger* tile; if already
filled, only amortization or occupancy moves `MfmaUtil`. Read the geometry + `VALUBusy` to pick.
(Both trade against spills/occupancy — see the register-spill ladder in `triage.md`.) On the Gluon
path a tile change is a **re-recovery** of the whole layout set, not a knob: request it as a
`resweep_request` (returned to tech_lead for GEAK's plain tuning) — `climb.md`.

### Rule: the coverage ratio disqualifies a config, it does not price one

The geometry above has a mirror case, and the ratio behind both has a hard limit on what it may be
used for. Write work coverage along a dimension as `reach(d) = warps_on_axis[d] × instr_extent[d]`
(in layout spelling, `warps_per_cta[d] * instr_shape[d]`) and read it against the tile extent along
`d`. `reach < extent` is the under-FILLED case above; `reach > extent` is **over-coverage**, and where
it sits decides what it costs:

- **Over-coverage on an input tile is redundant traffic; over-coverage on the output tile is a work
  multiplier.** The same ratio on the two tiles is two different findings, because the second wastes
  compute and not only bandwidth.
- **Use it to rule a configuration out, never to price one.** The ratio is cheap and static, which
  makes it a good disqualifier and a bad estimator: in one measured instance `reach/extent` predicted
  a **2.0x** multiplier where the ISA showed **12x per block**, leaving a 6x residual unexplained.
  **Do not pre-register a magnitude derived from the ratio** — the one band anyone checked against
  the ISA was ~6x wrong.
- **Basis trap, live in any script that varies the warp count.** Instruction counts read out of
  compiled assembly are **per warp**; the work multiplier is **per block**. Mixing the two produces a
  plausible-looking wrong factor, and it has forced a magnitude to be withdrawn twice,
  independently. **State the basis of every count in a table**, and multiply an assembly count by
  the warp count before comparing it against a per-CTA tile quantity. **Normalising to per-block also
  restores a free sanity check that the per-warp reading destroys: a block doing fixed arithmetic
  must issue a fixed count as the warps are redistributed.** In the same instance the per-warp column
  read `32 / 16 / 96` matrix ops across a warp sweep — inviting an explanation for a 2x between the
  first two that is not there — while per block it reads `32 / 32 / 384`, where the first two agree
  exactly and the anomaly is isolated to one configuration. Two configurations doing identical
  arithmetic that disagree per block mean either an instrument bug or a real anomaly and you do not
  yet know which; their agreement is what earns the count the right to be used at all. **This
  failure does not look like an error — it looks like a finding**: the per-warp column's 6x is
  large, clean and publishable, and it let a pre-registered mechanism covering only 2.0x of the real
  12x read as confirmed. The harder version of the same trap is a quantity the assembly cannot
  express at any scaling — a redundancy that lives *across* warps has no instruction to count, so an
  ISA diff returns an uninformative zero rather than a negative (`benchmark-hygiene.md ## A
  compile-only kill step is a falsifier, not a price`).

The under-covered direction has its own consequence: it is a legal repeat and carries no redundant
traffic, so **removing redundancy is a byte-count lever only where a tile over-covers.** An arm
priced on "fewer bytes moved *from de-duplicating coverage*" against an exact- or under-covering tile
is priced on a quantity that cannot move — which is separate from the byte-count levers that narrow
a dtype or add reuse, and those still apply.

### Rule: equal registers+occupancy but lower MfmaUtil => schedule/overlap gap

If a Gluon kernel has the **same registers+occupancy and the same `OccupancyPercent`** as the plain
target (**take the register and AGPR counts from the static AMDGCN dump**, not from `--pmc`: on a
measured gfx950 / ROCm-7 stack the profiler's AGPR column reads **structurally zero** even on kernels
whose descriptor plainly carries them, and its VGPR column reports a fixed fraction of the
descriptor's — the accumulator split does happen at run time, but the profiler does not observe it.
An earlier revision of this rule said the opposite; it was wrong in both directions) yet a **lower
`MfmaUtil`** and **more full-drain `s_waitcnt lgkmcnt(0)`** in the hot loop (`asm_loop_audit.py`;
threshold `perf_knowledge/hardware/data/thresholds.json` `confirm.full_drain_schedule_gap_pct`), the
gap is the **SCHEDULE** (operand-feed overlap), not registers/occupancy/LDS-count and not memory
bandwidth — bandwidth would show a saturated memory pipe and a dominant `MemUnitStalled` /
`stalled-on-L2`, and neither is present here. Call this reading `sub_resource=schedule-overlap`.

Fix, in the hand-written-first order (`../tile-programming/pipeline.md ## Authoring the overlap
yourself (the climb default on the Gluon path)`): (1) register-level prefetch, (2) an authored LDS
ring — gfx950: `async_copy` + `commit_group` / `wait_group`; gfx942 downgrade: sync staging, async
only 32-bit with `order=[1,0]` — (3) `warp_pipeline_stage` + the scheduling-model choice (layer 1.5,
gated on `num_warps >= 8`), then pace it (`../tile-programming/instruction-scheduling.md`). **Below
the parity gate** this gap is the `lost_pipeline` debt (`recover.md`); re-injecting plain's TTGIR
pipeliner (`reinject_ttgir_pipeliner`, `../tile-programming/pipeline.md ## Reproduce plain's software
pipeline on the Gluon path (the parity-recovery route)`) is **lowest priority**: a diagnostic that
measures the debt, or a last resort when the hand-written pipeline cannot reach parity
(`recover.md`, "Last resort: re-injecting plain's pipeliner"); its numbers are labelled
**"injected"**, never a win, and it is never used on an incumbent (already-Gluon) kernel. `num_stages`
is not a lever here — no 3.8.0 pass on the Gluon path consumes it. Do **NOT** record a scheduling
ceiling until the hand-written pipeline has been tried.

## Accuracy problems to distrust (profile + theory)

Both the measured profile and the theoretical budget can mislead; distrust these signals and
cross-check before acting on them:

- **Absolute time / TFLOPS drift under DVFS / shared contention.** Idle or partitioned cards
  downclock and light multi-kernel stages do not hold a ramped clock, so per-stage `do_bench` and
  coarse e2e swing run-to-run -> near-noise A/B and inflated single-sample "wins." Prefer
  **clock-insensitive** cycle / utilization counters as the discriminator (`benchmark-hygiene.md ##
  Shared / DVFS GPU timing`).
- **Roofline class != binding resource.** Aggregate arithmetic intensity assumes the compute is MFMA;
  for fused / VALU-heavy kernels (softmax-between-matmul, dequant) the kernel can be VALU- or
  dependency-stall-bound even when intensity says "compute-bound" -> profile the binding
  *sub-resource* before choosing a layer (`## Bound classification -> primary metric`,
  `../hardware/roofline-models.md`).
- **Cold / sub-ms / cached event-timing is unreliable.** ROCm event timing can contradict wall-clock
  / profiler on short or cached paths (`benchmark-hygiene.md ## ROCm timing`, `## Sub-ms Measurement
  Rules (also the gate's launch-floor evidence)`).
- **Clock-locking may be unavailable.** `rocm-smi --setperflevel high` can be "Not supported" on
  partitioned cards, so you cannot always force a stable clock — the discriminator must be
  clock-insensitive.
- **Budget mis-calibration.** A budget computed from drifting TFLOP/s or from planning-peak
  constants mis-sets the bound class; calibrate from clock-stable measured values after transcription
  (`budget.md ## Re-calibrate after transcription`, `../hardware/roofline-models.md ## Calibration`).
  A datasheet-denominated percentage ranks; it does not gate.
- **Profiler-reported registers.** AGPR/VGPR columns from `--pmc` are not the descriptor's (`### Rule:
  equal registers+occupancy but lower MfmaUtil => schedule/overlap gap`); A1/A3 come from the static
  dump and the cache metadata.

## rocprofv3 ATT (optional `--att`)

For "how well do MFMA/VALU/LDS overlap, and what steals the MFMA cycles" — the question the PMC
aggregate cannot answer — use rocprofv3 **ATT** (advanced thread trace), a separate pass from the
PMC counter pass. Run it through the `--att` mode of `profile_kernel.sh` (or `capture.sh`, which
auto-widens CU/SE and retries on `code:null`). When ATT completes, `hotspot_analyzer.py` runs on the
`ui_output_agent_*` directory under `$OUTPUT_DIR/att/` (same rocprof ATT input format as
tile-programming-flydsl; **not** the ncu CSV `hotspot_analyzer.py` in tile-programming-cutedsl):

```bash
cd <workspace>    # the GEAK workspace (KERNEL_PATH)
bash kernel_workflow/scripts/profile_kernel.sh <gpu_id> "python3 harness.py --profile" /tmp/gluon_prof --att
python3 kernel_workflow/scripts/kernel_tools/hotspot_analyzer.py /tmp/gluon_prof/att/ui_output_agent_* --topk 15 --mode both
```

ATT needs a debug-info build when source mapping matters. Visualize with `att_to_perfetto.py` (merge
several with `att_merge_perfetto.py`; timeline with `att_timeline.py`) → Perfetto.

**ATT decoder (required for `--att`).** `librocprof-trace-decoder.so` is **not vendored**: point
`ROCPROF_ATT_LIBRARY_PATH` at it (rocprofv3 `--att-library-path`). It is bundled on ROCm >= 7.13; on
older ROCm install the prebuilt `.so` during image build from
[ROCm/rocprof-trace-decoder](https://github.com/ROCm/rocprof-trace-decoder)
`releases/linux_glibc_2_28_x86_64/` into the image layer (or pass a fleet-provided path); never copy
it into host `/opt` (`scripts/README.md` Requirements). **Absent decoder = degrade** (drop ATT, record
it in `degraded[]`), not a failure. Wave-capture gotcha: if `code.json` has `code: null` (traced CU
caught no waves), widen the capture with `--att-target-cu 0 --att-shader-engine-mask 0xF`.

### ATT per-instruction ground-truth (the per-round VERIFY loop)

`hotspot_analyzer.py` gives source-line stall; two more tools turn the decoded ATT into the
ground-truth breakdown/pipeline (they SUPERSEDE `kernel_breakdown.py`'s INFERRED bubble):

- `kernel_tools/att_opclass.py <dir>/code.json` — per-op-class **active vs stall/bubble** (measured
  cycles).
- `kernel_tools/mfma_efficiency.py <dir>/se*_wv0.json <dir>/code.json` — MFMA cadence +
  **inter-MFMA CYCLE rollup** (softmax / lds / addressing / WAIT / SYNC) = the ground-truth ranked
  bubble-ownership (B1/B2). `kernel_breakdown.py --att <dir>` folds both into the one command.

Run it as a **per-round loop**: attribute -> attack the #1 bucket
(`../hardware/bound-class-signals.md ## Pipeline-breakdown -> lever (ATT ground-truth, per round)`) ->
re-run -> confirm THAT bucket shrank
(this closes the 验证 / verify step). Before spending an overlap lever, apply the
**structural-vs-schedulable** check (WAIT + `deep_mfma_analysis.py` run-length + occupancy; reorder
is inert at ~1 wave). Read everything **reference-free** (budget + floor-probe + absolute ceilings +
the previous round's ATT as the self-baseline; a vendor `--compare` peer is an optional diagnostic
upper-bound, never the gate) — `../hardware/bound-class-signals.md ## Reference-free ATT reading (no
gold / vendor kernel at runtime)`.

### rocprof-compute per-round (aggregate SOL + memory chart + occupancy limiter)

For the **aggregate "which hardware block is the ceiling / which memory level binds / is occupancy
resource-capped"** layer, `kernel_tools/rocprof_compute_probe.sh <name> <out> -- python <wrapper>`
runs rocprof-compute (`profile --no-roof` then `analyze --block 2 3 6`) and parses it
(`parse_rc.py`) to `rc_metrics.json`: System SOL block %-of-peak (MFMA/VALU/VMEM/LDS + IPC), the
memory chart (vL1D/L2/sL1D hit, BW, **L2-Fabric latency**), and the SPI occupancy-limiter
(Insufficient VGPR/LDS; empty = not resource-capped). Add `--roof` for the empirical hierarchical
roofline (MI200+; on gfx95x `--roof-only` needs rocprof-compute >= 3.6.0); its peaks are
`empirical@<tool-version>` denominators (`budget.md`). It is the Nsight-Compute analog and the
richest aggregate source.

- **Install / source of truth:** `ROCm/rocm-libraries` `projects/rocprofiler-compute/` (bundled from
  ROCm 6.2+; on ROCm 7.1 at `/opt/rocm/bin/rocprof-compute`). Its Python dependencies
  (`requirements.txt`) belong in the immutable image and are verified by fleet image-health before
  dispatch; a GEAK role reports a missing dependency as an environment blocker rather than running
  `pip install`. MI200+ only;
  roofline needs MI200+.
- **When to use:** once per optimization round (it is a **multi-pass replay** — re-runs the app per
  counter group; cost-control by filtering to the dominant kernel). For per-micro-tweak iteration the
  lighter `rocprofv3 --pmc` (`profile_kernel.sh --pmc` / `rocprofv3_safe.sh` + `parse_pmc.py`)
  suffices; ATT gives the per-instruction bubble. rocprof-compute complements — it does NOT replace
  ATT (no per-instruction attribution) or the static audit.
- **Gotchas:** argv-JSON is stripped by its re-exec -> use a **config-embedded wrapper**, not JSON on
  argv; if absent / PMC-blind (JIT), fall back to `rocprofv3 --pmc` + ATT + static.
- **Aggregation:** read `att/mfma_eff.txt` + `ir/asm_audit.txt` directly; together with the SOL they
  are the ATT rollup + the static audit + budget + timing into `round_<n>/record.json` (6-group
  schema) and the 3 must-have figures.

**Collection cost (ROCm 7).** `rocprofv3` defaults to a SQLite DB with a `rocpd` companion for
"**profile once, analyze many**" (generate CSV/OTF2/Perfetto post-hoc without re-running).
`rocprof-compute` collects counters in **multiple passes by default** (re-runs the app per pass);
single-pass iteration-multiplexing is upcoming. Budget profiling time accordingly (affects
`profile_kernel.sh` and `kernel_breakdown.py`).

## Asm hot-loop schedule audit (instruction + pipeline verification)

Counters and roofline classify the bound at **aggregate** granularity; the emitted assembly is the
**instruction- and pipeline-level cross-check** that the bound class and the pipeline are what you
think. Run it on the **plain** kernel's `.amdgcn`/`.s` **and** on the Gluon variant's — it is a
verification tool in **both** tiers, not a Stage-Gluon-only step.

**One-command entry:** `kernel_tools/kernel_breakdown.py <variant>.s [--pmc <csv_dir>]` merges this
static audit with the PMC busy/bubble table into one view + a **ranked bubble-ownership** stack
(feeds the ranked census); it auto-degrades to static-only with an INFERRED bubble when PMC is blind,
and `--compare a.s b.s c.s` produces a multi-kernel column table. The underlying static tool:

`kernel_tools/asm_loop_audit.py <variant>.s` (dump via `kernel_tools/dump_ir.sh`; strip per
`../tile-programming/compiler-contract.md ## IR dump workflow`) reports **signals only** (the
verdict is yours):

- **op-class instruction stream + histogram** (mfma / exp / lds-read / lds-write / global load-store
  / valu / scalar / waitcnt / barrier / nop) — an **instruction-level confirmation of which
  sub-resource binds**; cross-check it against the counter-derived bound class below (e.g. an
  MFMA-thin, valu/convert-heavy stream contradicts a "compute/MFMA-bound" reading);
- **`s_waitcnt` quality** — relaxed (`cnt>0`, consumer pipelined behind in-flight loads = good) vs
  full-drain (`(0)`, serialized handoff = conservative) — verifies whether the loop is **actually
  pipelined or serialized**;
- **producer<->consumer barriers per iteration** — excess-sync if it scales with pipeline depth;
- **`s_nop` + requested stall cycles** — an exposed **fixed-latency hazard** (e.g. MFMA-write ->
  VALU-read) hidden only by unroll/occupancy, never by reorder.

Use it as **bottleneck/optimization verification** at both ends of a step: on **plain** it confirms
the bound class and whether the auto-pipeliner already covered the loop (a serialized /
`s_nop`-heavy plain loop with explicit-control headroom strengthens the escalation case; a clean
relaxed-waitcnt stream argues stay-plain — `entry.md`); on **Gluon** the same read supplies the
**IR/asm signal** of three-evidence layer closure (`close.md ## IR acceptance (final)`). The deep
structural-vs-schedulable verdict and any sanctioned schedule fix live in
`../tile-programming/compiler-contract.md ## Auditing the hot-loop schedule`;
`asm_schedule_viz.py` renders the schedule.

### Reading the machine code of a JIT comparator you have no source for

When the kernel you want to read is JIT-compiled at runtime by a library or framework, there is no
object file to hand `llvm-objdump`, and PMC is blind to it (`## ROCm companion tools (AMD toolbox)`,
"Tool selection"). The compiled code is still on disk in that JIT cache, and the way to get it is to
**scan the cache as raw bytes** — not to deserialize an entry with the producing library's own
loader:

1. **Resolve the dispatch identity first** — the kernel name the run actually dispatched, tested
   rather than assumed (`benchmark-hygiene.md ## Every within-comparison stays green when both arms
   run the wrong thing`).
2. **Byte-scan the cache for that name to get the CANDIDATE SET — it is not a selector.** On a tuned
   library the kernel name is a function of the tile and the dtypes while the entry's directory hash
   is a function of the whole launch configuration, so many configurations compile a kernel of the
   same name. In one measured instance the scan returned **16 entries**, all with distinct digests in
   three size classes. **Do not break the tie by mtime.** Reading a cache entry does not touch its
   mtime, so "newest matching entry" is exactly the wrong rule in the case that matters — a dispatch
   served from a cold, older entry while something else compiled a newer one. In that instance two
   entries carried the current day's mtime seconds apart and the dispatch read the **older** one.
   Instead **make the dispatch name its own entry**: wrap `open` / `os.open` for the duration of
   **one** comparator call and record every path it opens. The answer is then observed rather than
   inferred, and it stays inside this section's own rule — the comparator is the thing under test and
   runs anyway; nothing in the cache is deserialised or executed.
3. **Lift the embedded artifact.** Inside the entry the compiled binary sits as an escaped byte
   string — an AMDGPU ELF. **Scan for the magic in every plausible spelling and report which one
   matched**, because the escape belongs to the producing library's serialiser and not to the ELF:
   the raw four bytes, a Python/C-style `\x7f`, and an **MLIR-style `\7F` — uppercase hex, no `x`**.
   Anything lowering through an MLIR `gpu.binary` / `#gpu.object` attribute uses the `\HH` form; in
   one measured instance the raw magic and the `\x7f` spelling each occurred **0** times in the entry
   and `\7FELF` occurred once, so a reader scanning only for `\x7f` concludes "no ELF here" and stops
   — a silent zero, not an error. Un-escape it and write it out as a `.elf`. **Take the image length
   from the ELF's own section table** (`e_shoff + e_shentsize * e_shnum`, extended by each
   non-`SHT_NOBITS` section's `sh_offset + sh_size`): the escaped literal may be followed by more of
   the serialised module, and nothing in the header states the total length.
4. **Disassemble it** with `llvm-objdump --mcpu=<arch>` (`--mcpu=gfx950` on the main line) and read it
   with the same loop audit you run on your own ISA (`amdgpu-disasm` / `asm_loop_audit.py`). **Assert
   the parse is non-empty before counting anything off it.** `llvm-objdump` on an AMDGPU ELF does not
   emit the host `addr: bytes mnemonic` layout — it emits `<mnemonic> <operands>   // <ADDR>:
   <encoding>`, mnemonic first with the address in a trailing comment — so a parser written for the
   host layout matches **zero** lines and every downstream count reads 0, which looks like a clean
   answer rather than a broken instrument. (Branch targets are printed symbol-relative,
   `<name+0xHEX>`, while the address column is absolute; subtract the function base before comparing
   them.)

**Read bytes; execute nothing out of the cache.** Deserializing an entry means running a loader over
data you did not write, and it ties the procedure to one version's internal structure. A scan for a
name plus a format magic depends on neither, and neither does watching which path the dispatch opens
— which is also why a cache's internal directory layout should not be written down anywhere: it moves
between releases, the name and the magic do not.

**Why this comes before any timing.** It turns "why is the comparator faster" from an attribution
argument into **side-by-side readings on the same axes**: register count, waves per workgroup, the
width composition of the loads, and the in-flight depth reached before the first `vmcnt` wait. Those
are the discriminants among the three standing candidates — **not wide enough / not enough in flight
/ not enough occupancy** — and not one of them needs a timer, a time attribution, or a latency
assumption.

One of those four reads is weaker than it looks on its own. **The width composition of the loads is
an issue-side quantity**; whether the access is wide is decided by the addresses adjacent lanes
present together, so a difference in load widths is not yet a difference in transactions and cannot
by itself support "not wide enough". Pair it with the lane-to-lane address delta before ruling on
that candidate: `../tile-programming/memory-path.md ## A load's width is issue-side; a transaction's
width is access-side`.

**The symbol names are a second, independent evidence chain.** An emitted ELF's symbol names
routinely encode dispatch parameters such as the blocking sizes. Where they do, they confirm the same
constants you read out of the library's API without sharing a failure mode with it — two chains that
cannot go wrong together. That is the first move of `benchmark-hygiene.md ## When the gate needs a
number you cannot read, bound it — do not guess it`, available here in two implementations instead
of one.

## Bound classification -> primary metric

| Bound class | Primary metric / next layer |
| --- | --- |
| compute: MFMA-issue | MFMA efficiency (`MfmaUtil`) -> pipeline / slicing / compiler contract |
| compute: VALU | `VALUBusy` (throughput, **not** `VALUUtilization`) -> reduce/fuse the VALU between matmuls (softmax/dequant/rescale) |
| compute: dependency-stall (latency C1) | dependency wait over Total (`VALUBusy` + `MfmaUtil` both low) -> **shorten the critical path** (fold critical-path scalars/scale off the chain — pre-scale Q, `exp2`) / raise occupancy; then break the dependency chain / interleave independent work (hand-written pipeline; or an LLIR pass under sanctioned co-design). Reduce-op-count and instruction-overlap are usually neutral here. |
| occupancy (latency C2) | issue wait over Total + SPI occupancy limiter -> raise resident waves / fill the GPU (slicing, registers, LDS) |
| memory (HBM) | achieved BW (C1) vs in-flight cap and in-shape ceiling (C3), only after C4 says memory binds -> memory path |
| lds | `ds_read` interval -> LDS layout (padding/swizzle) |
| register | VGPR/spill -> slicing |
| latency | pipeline coverage -> pipeline; or stay-plain wrapper/dispatch |

Within "compute," the binding sub-resource (MFMA-issue vs VALU vs dependency-stall) is not the
roofline class — read it from the profiler (System Speed-of-Light "Pct of Peak" and Wavefront Runtime
Stats above) before picking a lever. For a multi-stage / fused operator, profile **every
sub-stage**, not just the matmul — the dominant stage can be a non-matmul one (selection / reduction
/ dequant), and that is where the lever is.

**Classify → prioritize handoff.** Naming the bound class is only step "分类" (classify); the "优先级"
(prioritize) step reads the **ranked bottleneck stack** to decide *which* to attack first
(`../hardware/bound-class-signals.md ## Methodology spine (发现 → 分类 → 优先级 → lever → 验证)`). The
standard source of that stack is `kernel_breakdown.py` (the ranked bubble-ownership list — each
resource's share of the bubble / loop time, plus absolute numbers; see `## Output`). Enter via
`../hardware/bound-class-signals.md ## Bound-class decision tree (exhaustive, ordered, defaulted)`,
then descend the stack #1 → #2 → #3 (`climb.md ## Fallback ladder (escalate scope only after the
finer scope is exhausted)`).

**Bottleneck shift after a round.** Compare before/after on the same readings and write the shift
(`kernel_workflow/knowledge/profiling_guide.md` "Bottleneck Shift Analysis"):

```text
BEFORE: [bottleneck type] - [key metric value]
AFTER:  [bottleneck type] - [key metric value]
SHIFT:  [old] → [new] because [reason]
NEXT:   Target [new bottleneck] with [strategy]
```

## No-profiler fallback

If no profiler is available, label evidence `benchmark_only` and use roofline (shapes + measured
time), per-shape latency distribution, dispatch count, and `.ttgir`/`.amdgcn` inspection. IR signals
(`../tile-programming/compiler-contract.md`) still gate layer closure. On GEAK's ladder
`benchmark-only` is the floor, not a failure to fix — state plainly that no profiler was available.

### Canonical degrade ladder (one protocol — stop re-improvising)

Follow this exact order and record which rung you landed on in `metrics.json.degraded[]` (each entry:
`{"layer": <sol|warp_state|...>, "probe": "<exact cmd tried + error/signature>"}`):

1. **rocprof-compute analyze** — full SOL + memory chart + occupancy limiter.
2. ↓ if missing python deps (`plotext`/`astunparse`) / rc=4 / no-workload-dir → **`rocprofv3 --pmc`**
   (`parse_pmc.py`) for busy/util.
3. ↓ if ATT `code:null` / decoder rc=4 / decoder absent (`ROCPROF_ATT_LIBRARY_PATH` unset) → **ATT on
   a smaller shape**, or drop ATT and aggregate the rocprof-compute §7.2 (Wavefront Runtime Stats)
   buckets.
4. ↓ if kernel-trace sees 0 dispatches (graph replay), the box is PMC-blind (RDNA4 CDNA counters),
   a counter group times out in `rocprofv3_safe.sh` (exit 124), OR **the grid is too large to replay
   (present-but-SLOW / hangs)** → **native ISA (`.amdgcn` audit) + floor probe + boundary timing**.

The "present-but-slow huge-grid" case is distinct from "blind": the profiler is not absent, a very
large launch grid just makes the multi-pass replay impractical — treat it like a blind mode and
degrade, don't wait it out. Naming the rung is mandatory; a skipped tool WITHOUT a recorded probe is
non-compliant. The fixed `degraded[]` block makes the gate's A4 general rule work: a required
`metrics_read` field is excused only when its `source` layer appears here with a probe.

## Bound-the-win probe (before investing in a latency-hiding / pipeline lever)

The floor probe *localizes a stage*; the **bound-the-win probe bounds a lever's payoff** before you
spend rounds on it. Before a latency-hiding / overlap / prefetch / pipeline lever, **disable the
dependency or barrier that lever targets — accept a wrong result — and measure the resulting
ceiling**:

- ceiling ≈ current → the lever's max win is ~0 → close that lever (don't implement it). Example
  forms: an env-gated convert/barrier *elision* that pins TFLOPS; a `NOBAR` variant dropping a
  producer→consumer barrier.
- ceiling ≫ current → real headroom → the lever is worth building.

Run this bound-the-win probe FIRST for any lever whose card is `gating_law == occupancy_gates_latency`
(`climb.md ## 2. Reversed-intuition traps — read this once`, trap 6): on an occupancy-hidden kernel
the lever is a wash, and the probe is what tells you before you spend the round.

**What a probe number does NOT prove (anti-misjudgement).** "removing X → time T" is ambiguous: T is
either *unreached headroom* (the removing lever was never tried) or a *real ceiling* (every such lever
was tried, measured negative, each negative a proven wall). **A floor / bound-the-win probe bounds a
win; it NEVER, by itself, declares a ceiling** — a ceiling needs every ceiling-raising lever for
the #1 bound tried and measured negative, each negative a proven wall (the deep_engineer's closure
self-review checks exactly this before a close is recorded — `close.md ## Closure challenge (self-review
before a ceiling / keep-baseline / negative close)`; `close.md ## Stop conditions`).

## Plain front-end analysis (Stage-Plain)

Run this on **plain Triton**, before any Gluon work — it is the front end's analysis
(`front-end.md`), recorded here because it consumes the same profile. Output feeds the profile, the
budget, and the escalation gate (`entry.md`). In GEAK, GEAK's plain rounds own the plain search; this
section is the evidence the skill expects it to hand over.

### Record first

```text
repo / kernel file / kernel name / public wrapper / launch site
measured boundary: kernel-only | wrapper+kernel | full operator
target arch: gfx950 (main line) | gfx942 (downgrade) -- required; tools refuse a missing arch ;
  ROCm / Triton tag / PyTorch
workload class: one token from perf_knowledge/hardware/data/workload_models.json (models + aliases) — NOT free text
dtype + FP8/FP4 gate (gfx950 native OCP e4m3 vs gfx942 e4m3fnuz upcast risk)
correctness oracle / shape stream / fallback policy
```

**The class is a machine-consumed token, so spell it from the vocabulary.** The authoritative list is
`perf_knowledge/hardware/data/workload_models.json` (its `models` keys plus its `aliases`); this file
deliberately does not re-inline an enum, because a second copy is how the two drifted apart. Two
consequences to know before typing:

- **`other` is not a class.** It resolves as *unresolved*, which **withholds the structural axes**
  below rather than widening them — the opposite of what writing it usually intends.
- **A class that does not fit is a finding, not a formatting problem.** The archetype is the
  assumption most likely to be wrong, and `workload_models.json` carries a catalogue of the recurring
  ways a real kernel does not match the row sharing its name. When one applies, declare what the
  kernel moves as a manifest instead of stretching a row (`../workloads/intake.md ## 0. First: is it
  the archetype its name says?`).

### Plain-Triton worksheet [A]-[D] (from GEAK / Slide 24)

```text
[A] Primitives     : GEMM / reduction / scan / elementwise / attention / scatter?
[B] Shape regimes  : which shapes are common; how BLOCK_*, head, seq, K, batch vary
[C] Profile hotspots: which @triton.jit body owns time; HBM / L2 / MFMA / LDS / launch?
[D] Attack surfaces: 3-5 ranked hypotheses (direction, mechanism, expected metric move)
```

### Priority ladder (0-15; lower = higher priority)

```text
0   algorithmic rewrite (reduction tree, tiling, decomposition, split-K)
0.5 host/launcher OWNERSHIP (grid, workspace, config resolve, dispatch)  <- gates several of 0
2   fusion (adjacent kernels / elementwise / norm+quant)
5   memory/compute reorder (blocking, reuse, live-range, LDS)   <- main escalation trigger
6   shape-adaptive variants / visible dispatch
8   autotune / parameter search (BLOCK_*, num_warps, num_stages)
15  wrapper / launch micro-work only (lowest)
```

Note 0.5 against 15: **ownership** is near the top, **wrapper micro-work** stays at the bottom. The
ordering is a measured correction, not a preference: attempted early, the ownership layer has the
highest hit rate of any class on this ladder, and the ladder used to rank it last (`front-end.md ##
Why host/launcher is P0.5, and what it is not`).

Rule: prove the direction before sweeping parameters; autotune is refinement, not a substitute for
hot-path reasoning.

**Reduction-landing router (priority-0 structural signal).** Treat it as set when
`decision.structural_flag` and hoists the Layer-0 structural cards when the op is `multi_reduction`
(from `workload_models.json`) AND `metrics.structural.reduction_writeout_share` is above threshold
(dominant cost is a reduction write-out: atomic / RMW / materialize). The router only *points* at the
priority-0 decision — the counter cannot decide structure; **you make the fused↔split call from a
floor-probe, not from a prior.**

The structure decision is **symmetric and evidence-driven** — both directions carry a cost, so
floor-probe BOTH sides before committing: fused→split/defuse trades an atomic/RMW write-out for extra
launch(es) + recompute/re-read; split→fused trades that recompute/re-read tax for atomic contention.
When a floor-probe shows the reduction **write-out tail dominates**, going structural is the
**expected move, not a deferral** — three in-band (Tier-A) options: atomic-free two-pass, bf16
packed-atomic (`global_atomic_pk_add`; bf16 is lossy — verify vs oracle tol; MoE precedent
`moe_gemm_2stage.py`), and defuse by parallel axis (attention-bwd fused → separate dkdv / dq
kernels). Which one wins is decided by the floor-probe, not assumed — do not treat any prior
atomic-share number as universal.

### Hand-off to the gate

Analyze does not decide plain-vs-Gluon by itself. It produces the bound class (from this chapter) and
the budget gap (from `budget.md`); the escalation gate (`entry.md`) applies the criteria. The
Stage-Plain direction search and every stay-plain outcome live in `front-end.md` (organized by this
same 0-15 ladder).

**Worksheet [D] is the plain-stage BRANCH screen — FIVE STRUCTURAL AXES (+2 conditional) + ③ (layer-0)
cards.** The bound class is advisory; override it in the plan with evidence when the live profile
disagrees. [D] is anchored by **five workload-agnostic structural axes** worth considering for every
non-trivial kernel (`decision.prescan_candidates` with `axis_seed:true`), plus two that fire
conditionally. The five: **`computation_dag_reformulation`** (a different decomposition of the op
itself?), **`parallel_ownership_decomposition`** (a different parallel axis / who owns which
output?), **`kernel_boundary_topology`** (fuse vs defuse, and where the kernel boundaries fall),
**`intermediate_materialization_recompute`** (materialize an intermediate or recompute it?), and
**`reduction_aggregation_topology`** (where each reduction lands + live-accumulator count/dtype). The
two conditional ones — **`workload_partition_dispatch`** and **`cooperation_communication`** — fire
on multi-reduction and partitioned/communicating workloads.

> An earlier revision of this paragraph named three axes under a different set of ids. Anyone who
> worked from that list silently skipped **`intermediate_materialization_recompute`** and
> **`reduction_aggregation_topology`** — the materialize-vs-recompute question and the
> reduction-landing question, which are two of the highest-yield structural asks there are. The ids
> above are the ones the round engine actually seeds; take them from there, not from memory.

These are OPEN questions surfaced for EVERY non-trivial kernel — even when no named card matches — so
a composite / MLA / MoE kernel whose real structural opportunity has no card is still prompted to
consider one. The **③ layer-0 divergent cards** (`reduction_structure`, `packed_atomic_writeout`,
`grid_axis_defuse`) are **EXAMPLE tactics under these axes**. Good practice when you DO fan out an
axis: author a **`proposed:true` structural arm** (name the concrete tactic — a card OR a
first-principles rewrite like a single-live-accumulator two-pass — in
`candidate_directions[].proposed`, `arm_result.lever` stays the axis id) OR record a **backed N/A**
(`na_evidence`).

**Every axis owes a disposition; which disposition is yours.** All five axes above are surfaced for
every non-trivial kernel — plus either conditional one once its trigger fires — and each ends `fanned`
/ `rejected: <the reading that kills it>` / `deferred: <reason> + <priced lower bound>`. What is not
available is silence: an axis absent from the census leaves a record identical to one that was
considered and rejected, so the omission cannot be audited. Whether an axis is live remains a
judgement; record one explicit disposition and evidence for every axis the census raises.

Two other classes of divergent card are **NOT** [D] BRANCH work on the plain side: **①②
autotune-absorbable** cards (dispatch/wave-quant knobs + split_k →
`decision.autotune_absorbed_candidates`) are SWEEP: fold them into one `scripts/plain_autotune.py` (—
triton pack) config grid run, not per-round CLIMB directions; **④ `requires_anchor`** layer-4
pipeline/schedule cards (reinject_ttgir_pipeliner / warp_pipeline / manual_prefetch / attn_intra /
gemm_compiler) are **Gluon-loop-only** — their gate references a Gluon anchor ("vs plain" / Gluon
num_stages=1) that does not exist in the plain stage, so they never fan out here. Treat the [D] set
as the suggested best-of-N: **fan out one arm per axis + ③ card** you judge live (lite-profile each
per `orchestration.md`, keep the winner as the plain **target line**), do not iterate them
one-per-round in the tile loop. Fanning the arms out is a plain-front-end choice (in GEAK, tech_lead's
parallel specialist directions in the plain rounds); this CLIMB-only deep pack never fans out, so inside
the Gluon track the [D] set is worked sequentially (`orchestration.md`, `## Parallelism: bounded
measurements only, never a fan-out of the port`). Completeness — did you leave a ceiling-raising
structural bet untried before escalating? — is reviewed in the **closure self-review** (`close.md`), not
a hard gate. The one binding rule at the transition is Non-Negotiable #3 (`index.md ##
Non-Negotiable Rules`): the anchor is dumped from the pinned tuned-plain winner. "Prove the direction
before sweeping parameters" is guidance backed by that single pinned-config requirement.

### Verify stale guards

Check disabled fast paths / version guards that could replace the hot path (grouped vs per-head,
`tl.dot` vs elementwise, one-shot vs split/reduce). A guard removal that switches execution path is
its own direction — run correctness + a quick perf gate before treating it as the baseline.

## ROCm companion tools (AMD toolbox)

NVIDIA analog: the cutedsl skill's `references/companion-tools.md` (a sibling skill dir, or
`references/dsl/cutedsl/references/companion-tools.md` under the unified router). Operational tools
the agent invokes by phase — not skill dependencies.

### Tool selection (3 questions)

Pick the tool by the question, not by habit (ROCm Blogs "Introduction to profiling tools"):

| Question | Tool | Output |
| --- | --- | --- |
| **① Where should I focus?** (Amdahl / dominant kernel) | `rocprof-sys` (host+device timeline) / `rocprofv3 --kernel-trace` | timeline / hotspot ranking |
| **② How well am I using the HW?** (mem vs compute) | `rocprof-compute` (roofline + SOL) | roofline position, %-of-peak (an `empirical@<tool-version>` denominator) |
| **③ Why this performance?** (root cause) | `rocprofv3 --pmc` / `rocprof-compute` (block SOL, memory chart, SPI) | per-unit counters, occupancy limiter |

Then hand ②/③ evidence to `../hardware/bound-class-signals.md ## Bound-class decision tree
(exhaustive, ordered, defaulted)`. For a single nominated kernel (this skill's usual entry) ① is
often already known; still run ② before deep counters. `rocprof-compute` is the Nsight-Compute
analog and the richest single tool when installed; when it is absent or PMC is blind (JIT kernels),
fall back to `rocprofv3 --pmc` + static ISA (`## Profiler-capability preflight (verify ONCE — do not
assume PMC is visible)`).

### Profiling / inspection

| Tool | Use | Phase |
| --- | --- | --- |
| `rocprof` / `rocprofv3` | PMC counters, roofline | profile |
| `rocprof-compute` | block-level SOL + memory chart + SPI occupancy-limiter + empirical roofline (Nsight-Compute analog) | profile (per round) |
| `rocprof --hip-trace` | timeline / launch | profile (eager boundary) |
| `amdgpu-disasm` / `llvm-objdump -d` | ISA hot-loop audit | profile / IR |
| FlyDSL `FLYDSL_DUMP_IR=1` | MLIR → ISA chain | profile |
| TileLang `plot_layout.py` | layout audit | smem layout |

### Correctness

| Tool | Use |
| --- | --- |
| `compute-sanitizer` (HIP port) | memcheck where available |
| Oracle `allclose` in harness | mandatory per dtype |

### Environment

| Tool | Use |
| --- | --- |
| `rocm-smi` | clock / power |
| `rocminfo` | agent arch (`gfx950`, `gfx942`, `gfx1201`) |
| `scripts/gpu_identity.py` (GEAK) | GPU identity / Instinct product mapping (`benchmark-hygiene.md`) |

### ISA PDFs

| Doc | URL |
| --- | --- |
| CDNA4 ISA | [AMD Instinct CDNA4 ISA (PDF)](https://www.amd.com/content/dam/amd/en/documents/instinct-tech-docs/instruction-set-architectures/amd-instinct-cdna4-instruction-set-architecture.pdf) |
| CDNA3 ISA | [AMD Instinct MI300 / CDNA3 ISA (PDF)](https://www.amd.com/content/dam/amd/en/documents/instinct-tech-docs/instruction-set-architectures/amd-instinct-mi300-cdna3-instruction-set-architecture.pdf) |
| RDNA3 ISA | [RDNA3 shader ISA (PDF)](https://www.amd.com/content/dam/amd/en/documents/radeon-tech-docs/instruction-set-architectures/rdna3-shader-instruction-set-architecture-feb-2023.pdf) |
| RDNA4 ISA | [RDNA4 shader ISA (PDF)](https://www.amd.com/content/dam/amd/en/documents/radeon-tech-docs/instruction-set-architectures/rdna4-instruction-set-architecture.pdf) |

### SKU datasheets (planning peaks)

Single numeric source: `perf_knowledge/hardware/data/sku.json`; the markdown sheets explain it.
Datasheet peaks rank; they never gate (`budget.md`).

| Layer | Files |
| --- | --- |
| CDNA4 gfx950 | `../hardware/amd-cdna4-skus.md` — MI350X, MI355X |
| CDNA3 gfx942 (downgrade) | `../hardware/amd-cdna3-skus.md` — MI300X, MI308X, MI325X |
| RDNA3 gfx1100 discrete | `../hardware/amd-rdna3-skus.md` (not supported by GEAK; reference only) |
| RDNA 3.5 gfx1151 APU | `../hardware/amd-rdna35-skus.md` — AI Max+ 395 / PRO 495 |
| RDNA4 gfx120* | `../hardware/amd-rdna4-skus.md` |

`kernel_tools/calc_perf.py roofline --sku MI350X` vs `MI355X` — same ISA, different clock basis and
peaks; likewise `MI308X` vs `MI300X` (gfx942) — same ISA, different ridge/CUs. `calc_perf.py` refuses
a dtype the SKU row does not carry (no fp16 fallback).

### Code anchors (RDNA / CDNA)

| Family | Upstream (git) |
| --- | --- |
| FlyDSL RDNA kernels | [ROCm/FlyDSL `kernels/`](https://github.com/ROCm/FlyDSL/tree/main/kernels) (`rdna3_f16_gemm.py`, `rdna_f16_gemm.py`, `rdna_fp8_preshuffle_gemm.py`) |
| Gluon AMD API | [triton-lang/triton `gluon/language/amd/`](https://github.com/triton-lang/triton/tree/main/python/triton/experimental/gluon/language/amd) |
| TileLang ROCm | [tile-ai/tilelang `tilelang/rocm/`](https://github.com/tile-ai/tilelang/tree/main/tilelang/rocm), [`testing/python/amd/`](https://github.com/tile-ai/tilelang/tree/main/testing/python/amd) |

## Output

```text
profile/<round>/{trace.csv, att/, counters.txt}        # in GEAK: profile_kernel.sh's
                                                       #   <output_dir>/profile_report.txt + subdirs
gluon_anchor_metrics.json (after transcription) ; per-round profile JSON
```

Record a **ranked top-3 bottleneck list with absolute numbers** in the per-round profile JSON (not
just "the binding class"). The **standard source** of this ranked stack is `kernel_breakdown.py`
(static ⊕ PMC) or, for a full per-round record, reading them together — rocprof-compute SOL
(`rocprof_compute_probe.sh`, optional) ⊕ the ATT inter-MFMA rollup ⊕ the static audit ⊕ budget +
timing into `round_<n>/record.json` (6-group schema) plus the 3 must-have figures (bubble
decomposition, ranked stack, round trend). The layer loop descends this stack: when #1 hits its floor
it attacks #2/#3 (`climb.md ## Fallback ladder (escalate scope only after the finer scope is
exhausted)`, ladder step 2a), so the rest of the census must be kept, not discarded once #1 is named.

**The `round_<n>/record.json` + 3 figures are MANDATORY every round** — declared in `## 3.1 Required
evidence — the four dials, every round` and enforced by the Record gate (`records.md ## 8. Final
Delivery Record (1:1 with final_report.json)`). The same content goes into GEAK's round record and
profiling summary (`profiler_used` + every degrade named). When a tool is blind,
name the degrade path in `caveats[]`; never silently skip the record.

> **Counter vs timing after a fix.** A per-access *rate* counter (`LDSBankConflict`,
> `VALUUtilization`) can stay flat while the kernel gets faster — a swizzle that de-serializes
> conflicting accesses cuts cycles while the *fraction* that conflict is unchanged. So **timing at the
> contract boundary is the keep/revert arbiter; the counter is the discovery tool** (it localizes the
> bottleneck), never the overriding accept/reject signal in three-evidence closure.

## Sources

Merged into this chapter (old paths, relative to the skill root): `references/phases/profile.md`,
`references/phases/analyze.md`, `references/execution-locus.md`, `references/rocm-companion-tools.md`,
`references/method-reference.md` "3.1 Required evidence — the four dials, every round" and "3.2
Evidence layers: the floor, and the full read", and the compressed copies of 3.1/3.2 in
`tile-programming-gluon.md` (unioned: A1 wg/CU conversion + LLVM `; Occupancy` caveat, A2 vacuous
VGPR half / 345 spill slots, B3 keep `s_waitcnt` in the diff, D3 knob-hash row). Interpretation rules
follow `kernel_workflow/knowledge/profiling_guide.md`.

Rewritten / demoted (GEAK infrastructure and the conflict rules win):

- Profiler entry: the pack's `scripts/profile_kernel.sh <harness> <out> [gpu] [warmups] [--att]`
  (which exported `HIP_VISIBLE_DEVICES` inline) → GEAK `kernel_workflow/scripts/profile_kernel.sh
  <gpu_id> <cmd> <out>` with optional `--pmc` / `--att` modes, under `gpu_lock.sh`; the pack's
  counter/derived/SPI defaults are kept as the `--pmc` content. Parsers/wrappers cited at
  `kernel_workflow/scripts/kernel_tools/`.
- Report-reading heuristics: "Dependency Wait (memory) vs Issue Wait (compute)" and "Dependency Wait /
  Active > 3x => memory-bound" → DROPPED as written; replaced by `kernel_workflow/knowledge/profiling_guide.md`: ratios over
  Total, dependency wait = latency C1, issue wait = latency C2, memory-bound only on high HBM/VMEM busy
  plus a C4 floor probe; compute-bound read on busy counters (`VALUBusy`/`MfmaUtil`), not
  `VALUUtilization`.
- D1 "stable-min **and** median" from `ab_bench.py` → acceptance timing from `harness_lib.py`
  (median, read-evict, fresh process per leg, same-window baseline, `MIN_IMPROVE=2%`); `ab_bench.py`
  demoted to search/screening (median + spread, min supplementary).
- "Rule: equal registers+occupancy but lower MfmaUtil": the fix "re-inject plain's TTGIR pipeliner
  (Route-1)" → hand-written pipeline first; re-injection demoted to a below-parity diagnostic / last
  resort with numbers labelled "injected", never on an incumbent.
- rocprof-compute dependency install "fleet image-health" kept; the "argv-JSON" and multi-pass
  hazards deduplicated (stated once in `### rocprof-compute per-round …`).
- ATT decoder: "pass a fleet-provided `--att-library-path`" generalised to `ROCPROF_ATT_LIBRARY_PATH`;
  absent decoder = degrade.
- Counter-slot overflow: "SIGABRT at 4 on some gfx942 boxes" kept as an instance; the behaviour
  (replay / abort / hang) is stated as version-dependent and handled by `rocprofv3_safe.sh` timeout +
  degrade.
- Execution locus: `locus.sh` / `TILE_KERNEL_CONTAINER` scoped to a kernel in a separate container
  (optional); its path-contract heading renamed from the run-mode suffix to `(separate container)`
  (unpinned; the one citation updated).
- Run-mode split (GEAK-embedded vs the pack's `toolctl` spine / upstream gluon-direction agent) removed:
  profiling is `profile_engineer`'s (baseline, re-profile) and the deep_engineer's (in-loop); the
  closure-skeptic agent → the deep_engineer's closure self-review; subagent fan-out of the [D] set → the
  plain rounds' parallel directions.
- Example arch ordering: gfx950 first, gfx942 as downgrade (ISA PDFs, SKU sheets, `rocminfo`).
- Analyze "workload class" path → `perf_knowledge/hardware/data/workload_models.json`; target arch is
  required (no silent default).

Dropped: none beyond the two heuristic lines above (true duplicates between the method-reference and
`tile-programming-gluon.md` copies of 3.1/3.2, and between `phases/profile.md` and
`rocm-companion-tools.md` on ATT/rocprof-compute, were merged into one statement each).
