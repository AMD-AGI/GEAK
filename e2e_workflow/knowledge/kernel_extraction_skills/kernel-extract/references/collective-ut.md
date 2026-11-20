# Multi-rank UTs (collectives and fused multi-rank kernels)

A kernel is multi-rank when one invocation needs several GPUs at once: a collective
called through a communication library (all-reduce, all-gather, reduce-scatter, a
custom all-reduce), or a fused kernel that exchanges data with the other ranks
INSIDE the kernel through peer memory (a "mono kernel": a whole decoder layer, its
all-reduces included, in one persistent launch). It stays an extraction target. A
single-GPU runner cannot exercise it; the generated UT runs it on its whole GPU group
with one process per rank. Preserve the serving world size, dtypes and dispatch cases.

## Decide and declare

Collect the evidence before writing the task: the serving tensor-parallel size, the
trace (every rank launches the same kernel at the same position of a step), and the
source (IPC handles `hipIpcGetMemHandle` / `hipIpcOpenMemHandle`, peer pointers,
symmetric memory, system-scope stores and flags, communicator calls). A kernel whose
name does not look like a collective can still be one. Declare it in `meta.json`:

```jsonc
"capabilities": {
  "stateful": false,
  "multi_rank": {
    "world_size": 4,                        // == run/requested_kernel.json required_gpu_count
    "comm": "peer_memory",                  // or "collective_api"
    "inputs": {"hidden_states": "replicated", "w_qkv_shard": "per_rank", "kv_pages": "per_rank"},
    "evidence": {"serving_tp": 4, "trace": "run/selection/...: 4 ranks launch mono_layer per step",
                 "source_symbols": ["hipIpcOpenMemHandle", "__hip_atomic_store system scope"]}
  }
},
"timeouts": {"setup_s": 900, "command_s": 1800, "total_s": 3600}
```

`required_gpu_count` in `run/requested_kernel.json` (and in the result) must equal
`world_size`; the delivery gate refuses a mismatch. A collective without an explicit
`required_gpu_count` is deferred, not optimized: no GPU count is assumed for it. A
kernel declared single-GPU whose `kernel_src/` shows such symbols is refused unless
`capabilities.single_rank_justification` explains why.

## Live implementation and editable source

Resolve the communicator or peer-memory runtime the serving stack actually uses,
including backend aliases (a SGLang deployment may use AITER's communicator rather
than the similarly named SGLang class). Inspect the actual call and captured device
symbol. Establish one native eager/graph replay before expanding the task; reuse this
proof when it already exists (every UT run repeats it: the native pools run every
case first, and a native failure stops the UT with exit 3). Copy the native
implementation and required local headers, with original paths and hashes
(`meta.kernel_extraction.source_origins`). An installed Python wrapper or a compiled
binary alone does not prove native editability. Reuse compatible build settings from
the deployed stack, resolving relocated include paths against the current container.
Record dependency paths, compiler flags and the built library's hash; keep generated
libraries under the task with distinct identities per source content. Bind the
candidate through the same deployed callable/overlay contract as other tasks. A
prototype that directly installs a local native module is useful source proof, but
the final task still needs baseline isolation and candidate binding (the UT's
`legs_differ` gate checks both). If an unchanged communicator constructs native state
for the candidate call, verify that its ABI and state layout remain identical.
Preserve IPC allocation, registration and destruction lifecycles; numerical inputs
are not the complete collective state.

## cases.py for several ranks

The generated UT starts `world_size` rank processes per leg (`leg_runner.py` serve
mode): each sets `RANK`, `LOCAL_RANK`, `WORLD_SIZE`, `MASTER_ADDR`/`MASTER_PORT`,
selects its device (`LOCAL_RANK`), runs `setup` once and then the same commands in
the same order. Hooks in addition to the single-GPU ones:

- `setup(h, meta, ctx)` -- optional: build this rank's state (framework distributed
  state, TP group, weight shards, IPC buffers). Without it the UT initialises
  `torch.distributed` (NCCL on GPUs) with the `setup_s` timeout and selects the
  device. `teardown()` releases what `setup` built.
- `ctx` (`h.rank_context()`, also passed to `make_inputs(rng, ctx)`): `rank`,
  `world`, `local_rank`, `data_rank`, `generator(seed)`, `rank_rng`. Build every input
  that differs per rank -- weight shards, KV pages, per-rank activations -- from
  `ctx.data_rank` / `ctx.rank_rng` / `ctx.generator(seed)`, never from
  `os.environ["RANK"]`: the `rank_skew` control makes the last rank build the previous
  rank's data and the UT must reject that rank. Replicated inputs use the plain `rng`
  (the same on every rank). Keep numerical inputs distinct across ranks; identical
  inputs cannot catch a kernel that reads another rank's buffer.
- Compare every rank's output with an independent sum/reference: give the eager cases
  a `ref` computed without the communicator (for an all-reduce, the sum of every
  rank's input); the native reference run checks it on every rank.
- Every rank must issue the same collectives in the same order: hooks may not branch
  on the rank, and case lists, iteration counts and timing cases are identical on all
  ranks.
- `capture_context()` -- the framework's graph-capture context, when the
  communicator registers captured buffers on exit (replaying before that exit is
  invalid). This follows the communicator's graph-registration lifecycle: the UT
  enters it around every capture, for the replay checks and the timing alike.
- `align(h, meta, ctx)` -- optional, for a kernel with its own cross-rank sync: a call
  that synchronises the ranks on the device WITHOUT touching the tested data; used
  instead of the UT's one-element all-reduce (below).

## A mono kernel launched several times per step

A mono kernel is often launched several times in one deployed step, e.g. once per
layer, and each launch's parameters (`layer`/`nlayers`, `first`/`last`, ...) decide
the path it takes: whether it returns early, whether it advances an epoch or step
counter, how its tag is composed. The MiMo mono layer kernel, for example, tags its
messages 64 x epoch + layer + 1, and a launch with `layer != nlayers - 1` returns
early without advancing the epoch: only the last launch of a step advances it, while
a deployed step runs the other path in all its other launches. A UT whose every
launch is a step of its own (`layer=None`, so `nlayers` is 1) takes the last-layer
path only. For such a kernel:

- At least one replay group records, in ONE capture, >= 2 launches with the
  deployment's per-launch parameters, covering every path a deployed step contains:
  first, middle (when the step has one) and last. This group declares `carry_state`
  and replays >= 2 consecutive steps.
- `capabilities.stateful_tiers` names this transition inside a step, e.g. "a
  non-last-layer launch does not advance the epoch, the last one does", and
  `sequence_cases` covers it with steps of several launches.
- Every launch's output is compared, either directly or through the output of a later
  launch that reads it; say which one (e.g. in `meta.oracle_coverage`).
- Launches that each form a step of their own (`layer=None`) do not replace any of
  this.

The generator cannot tell what these parameters mean and does not check this; see
also [task authoring](task-authoring.md), "Graph replay groups".

## What the generated UT does with several ranks

- **Pools.** One pool of rank processes per leg: native references, native repeat and
  candidate for correctness, native and candidate for timing; each pool has its own
  rendezvous port (retried on a taken port) and its own control group over gloo. A rank
  that fails ends its pool at once and the pool's processes die with the driver.
  Process-group initialization is bounded by `setup_s`, and every rank's result is
  collected and judged: a launcher that exits cleanly, or rank 0 alone, never passes. A
  kernel that reads/writes peer memory (`comm: "peer_memory"`) never has two pools
  issuing GPU work at the same time (setup included): persistent kernels of two pools
  on the same GPUs would wait on each other forever.
- **Correctness.** Every check of the single-GPU UT -- eager, random draws, the graph
  replay of every replay group after `fill` changed the rank's inputs in place,
  sequence -- runs on every rank against that rank's recorded native outputs; a rank
  that fails fails the UT. Graph capture is agreed: all ranks capture or none (a graph
  deployment then FAILS).
- **Timing.** Each timed sample is aligned before its start event: cache flush
  (`harness_lib.flush_cache`) → device sync → gloo barrier that agrees a release time
  → busy-wait to it → a short GPU spin on the rank's stream (default 200 µs,
  `meta.timing.device_spin_us`) → a one-element all-reduce on the pool's device group
  (or `cases.align`) → start event → kernel → end event. The spin covers the time each
  rank needs to queue the rest, so the all-reduce releases every GPU together. A
  sample's time is the maximum over the ranks; the case time is the median over the
  samples; `rank_ms` keeps each rank's median. All ranks must use the same timer
  (graph replay or eager), and both legs too. The receipt (`GEAK_TIMING_RECEIPT`,
  `run/ut_result.json` `receipt`) carries `world_size`, every case's `rank_ms`, each
  leg's `timer` (the graph mode it actually used) and this `alignment`.
- **Environment.** Rank processes inherit the caller's environment (user site and
  `PYTHONUSERBASE` included), get `HIP_VISIBLE_DEVICES` and `CUDA_VISIBLE_DEVICES` set
  together to the leased group (they must agree or ROCm aborts), and never inherit a
  deployed `ROCR_VISIBLE_DEVICES` / `GPU_DEVICE_ORDINAL`. With two pools alive on the
  same GPUs the UT sets `AITER_CUSTOM_AR_RAW_INPUT_POOL=1` for them (a test-harness
  setting, recorded in the receipt; a deployment that already sets it keeps its value).
- **Time limits.** `meta.timeouts`: `setup_s` per pool start, `command_s` per pool
  command, `total_s` for the whole run (it also bounds the group lease; defaults 900,
  1800 and 3600 s). `setup_s` and `command_s` must be above 0 (`generate_ut.py check`
  refuses 0). A run that exceeds one stops cleanly with exit 2 and the reason.

## Lease and run

Lease `SERVING_GPU`, the serving GPU set, as ONE group and run the UT directly -- it
starts its own ranks; never wrap it in `torchrun` or another launcher, and never nest
leases:

```bash
cd "$TASK_DIR"
GEAK_GPU_RUN_TIMEOUT=<meta.timeouts.total_s> \
  bash "$SHARED_TOOLS/../../kernel_workflow/scripts/gpu_lock.sh" "group:$SERVING_GPU" -- python3 unittest.py
# verification under the same kind of lease
GEAK_GPU_RUN_TIMEOUT=<several x total_s> \
  bash "$SHARED_TOOLS/../../kernel_workflow/scripts/gpu_lock.sh" "group:$SERVING_GPU" -- \
  python3 "$SHARED_TOOLS/verify_ut.py" --task .
```

Source-mutation runs (`python3 unittest.py --mutation <leaf>`) lease the same way.
Better, run the UT, `verify_ut.py` and every mutation as the one chain of
[SKILL.md](../SKILL.md) step 5 inside a single group lease whose
`GEAK_GPU_RUN_TIMEOUT` covers the whole chain. The group lease stops a hung run after
its run timeout (900 s when unset) and reaps what it left. If fewer GPUs are visible
than `world_size`, the UT stops with an environment error (exit 2).

## Acceptance

Prove native source execution with a reversible numerical change in the selected
device implementation, such as changing its reduction result. Every rank must reject
the returned values: record each rank's row from `correctness.by_rank` of that
mutation run's `run/mutations/<leaf>/ut_result.json`
(`ranks: [{rank, launch_succeeded, numerical_mismatches}]`, see
[source records](source-delivery-records.md)); an import, IPC or compilation error is
not numerical rejection. Restore the original source for final validation. Record
actual symbols and launch configurations per case and rank, including dispatch
switches that require different reference kernels.

One-case API, editable-source or negative-control success is an intermediate result.
Completion requires the generated UT passing on all ranks, `verify_ut.py` `PASS`
(its `rank_skew` control included), all declared positive-mass cases on both legs,
baseline/candidate identity, observed-call weights, per-rank evidence and a final task
report. Preserve the original whole-workload denominator when a collective case
remains uncovered.
