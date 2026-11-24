# Writing a mono_layer UT (multi-rank monokernel layer)

A `mono_layer` task tests ONE layer launch of a persistent whole-layer monokernel
(sgl-amd-mono: Muse Spark target / draft, ...; vLLM `--monokernel`: DeepSeek-V4.1-Flash) on every
TP/EP rank, with real in-kernel peer exchanges. `scaffold_ut.py new --family mono_layer ...` writes everything except the model:

| file | who writes it | what it is |
|---|---|---|
| `rank_harness.py` | scaffold (copy of `ut_contracts/mono_harness.py`) — **never edit** | process topology, command loop, poisoning, cache snapshot / stray counting, graph replay, interleave, `rank_chain` timing (`Chain`), JIT cache key (`leg_env`) |
| `rank_model.py` | **you** (`templates/rank_model.py.tmpl`) | `class RankModel`: weights, KV pools, page tables, the launches, the runner's buffers |
| `cases.py` | scaffold; you fill only the call-site docstring | manifest rows → operands → harness; `ORACLES` generated from the contract's `oracles` |
| `run/ut_cases.json`, `meta.json` | you | the observed cases; negative controls (stubs generated per required category), evidence |

Worked examples, both verified on GPU with every gate including the native oracle floor:

- `references/examples/muse_spark_target_rank_model.py` — Muse Spark target layer (TP8/EP8,
  routed MoE): one launch a layer, merged + split + native paths; native = sgl-amd-mono's
  per-stage torch references, teacher-forced, near-tie routing aligned.
- `references/examples/muse_spark_draft_rank_model.py` — Muse Spark DFlash draft layer (TP8,
  dense): a layer is several device launches and the deployed path has NO merged launch (split is
  the baseline's own code, so correctness rests on native); the call goes through the wrapper
  `_Draft.forward` (a per-layer runner proxy); native = SGLang's own `DFlashDecoderLayer` through
  `_Draft._attention_reference / _reference`, built at TP 1 outside the server
  (`get_context().override_server_args(...)`, `get_parallel().override_permanently(tp_group=None)`,
  `forward_context.set_forward_context(ForwardContext(attn_backend=...))`).

- `references/examples/dsv41_flash_target_rank_model.py` — DeepSeek-V4.1-Flash target layer (vLLM
  `--monokernel`, TP2, routed MoE): a "launch" is one `DSV41MonoLayer.forward` (K1 + K2 device
  launches); the stream is the mHC residual `[rows, 4, 5120]` with its seam mixes (section 2b);
  fp8_ds_mla KV records through `kv_bytes` / `kv_decode`; a read-only compressed cache with a
  top-512 `topk_indices` input. A step is the five served layer types (r0 fold 0, r2, r1, r2 index
  source, r1 index source; an index layer's `select` copies a precomputed top-512, standing in for
  vLLM's indexer); `launch_waivers` cover the one build it does not reach. Paths: deployed;
  `split` = the same deployed call on the baseline overlay (DSV41MonoLayer has no per-stage mode:
  the Muse draft precedent); `native` = vLLM's own non-mono decode op chain as vLLM's
  `tests/models/test_dsv41_mono_numerics.py` runs it (mHC seams, the ROCm attention ops, the
  all-reduce over gloo, `DeepseekV4MoE` built at TP 1 on the rank's shard with vLLM's model-parallel
  state of TP size 1 over the harness's gloo world), teacher-forced per launch, near-tie routing
  aligned on the FFN input K2 used (scratch `normed`). `ids` stays `not_exposed`. Deployed path
  GPU-verified 2026-11-13 (non-index layers, every correctness-mode gate, 6/6 controls, rank_chain);
  the layer types, split and native (written 2026-11-16 without a GPU) GPU-verified 2026-11-17:
  every gate incl. the native floor (baseline min SQNR 31.9 dB against 20), 6/6 controls, A/A 1.00
  (`references/mono_models.json` `oracles_verified`).
- `references/examples/dsv41_flash_draft_rank_model.py` — DeepSeek-V4.1-Flash DSpark draft (the
  draft runner: 128 experts / top-3, `swa_width` 192): a step is the draft's 3 layers (the first
  fold 0 on the broadcast embedding), 5 rows a request with the non-causal block keys of vLLM's
  DSpark SWA metadata, one pool (no compressed cache). Helpers and the native chain are the
  target example's verbatim (the DSpark `DeepseekV4MoE`). GPU-verified 2026-11-17: every gate incl.
  the native floor (baseline min SQNR 30.6 dB), 6/6 controls (the model's own: no tuned build runs in
  the draft, see section 3), A/A 1.00.

Copying their structure is expected; copying their numbers is not — every constant must come from
your model's config and server evidence. Which oracles a model's example covers is
`references/mono_models.json` `oracles_ready` (GPU-verified) and `oracles_unverified` (implemented, no
GPU run yet: kept in the contract with the family's provisional bounds, and the request says so); a
family oracle no example covers is dropped from the task's contract by the request
(`scaffold.contract_overrides`, with a why), never left for you to invent.

## 0. Requesting a mono_layer task

The UT request author ([request-authoring.md](request-authoring.md)) writes a mono_layer task from
`references/mono_models.json`, one task per model whose `kernel` pattern the selected kernels match
(a profiled group holding a target and a draft monokernel becomes two tasks; its other kernels stay
a task of their own):

- `label`, `scaffold.target_callable` and `scaffold.bind_module` from the model;
  `scaffold.bind_file` = `kernel_src/<bind_module as a path>/__init__.py` (the whole package binds);
- `scaffold.family: "mono_layer"`, `scaffold.attrs` from the model, `scaffold.axes` = the model's
  `axes` plus `world` (the deployment's tp; also `requested_kernel.required_gpu_count`). Optional
  family axes (`topk`, routed_moe only) are pinned only where the model has them;
- `scaffold.phase_regimes`: `{"DECODE": {"cuda_graph": true}}` for a graph-launched decode;
- case axes from the model's `case_axes` for each phase the task runs, with the serving-arg
  placeholders filled from the launch args (`widths_from` says where the widths come from);
- `scaffold.contract_overrides` (`[{path, value, why}]`): every family oracle not in the model's
  `oracles_ready` / `oracles_unverified` dropped from `oracles` (and a `reference` that named it
  moved to a kept oracle), and every output in the model's `not_exposed` given
  `present_when: "False"`, each with the reason; `scaffold_tasks.py --check` refuses an override
  without a why;
- `launch_waivers`: the model's rules (`[{pattern, why}]`, builds the worked example does not
  reach) as they are; `scaffold_tasks.py` expands them over `expected_launches` into
  `meta.ut_contract.launch_waivers` (the lint refuses a waiver without a why or of a kernel the
  evidence does not expect);
- `negative_controls`: the model's GPU-verified controls (section 3) as they are;
  `scaffold_tasks.py` writes them into `meta.ut_contract.negative_controls` in place of the
  scaffold's FILL stubs;
- `manifest.replay_sigs`: see section 4 (one width).

Check every copied value against the model's config and the serving launch.

## 1. Read the deployed caller first

Record the call site (file, line, exact call, the order of calls within a step, every rank's
calls) in `cases.py`'s docstring before writing anything else.

- Muse target: `sglang/srt/monokernels/runtime.py` (`model.forward` = `built.forward`) →
  `sgl_amd_mono/muse_spark/spec.py::_Built._backbone` (ef38b0e: :210), which calls
  `r.launch(i=...)` per layer (:229) and `r.end_step()` after the step (:239). The model file
  `rocm_muse_spark/model.py` never touches the runner: follow the monokernel runtime, not the
  model's own forward.
- Muse draft: `draft.py::_Draft.forward` → `launch_attn / launch` (v0 `launch_layer`).
  The draft's call() must go through `_Draft.forward`.

`RankModel.launch()` must make the deployed call with the deployed argument types; anything else
tests a different code path.

## 2. RankModel rules

1. Fill only `RankModel`; do not change the topology. One process a rank, the subject's real
   peer buffer over the `group` the harness passes.
2. Weights: TP / EP shards seeded per rank; replicated tensors identical on every rank; no norm
   weight all ones, no two norms equal (assert it); q_norm ≠ k_norm where the model has weighted
   q / k norms (Muse's are weightless: nothing to vary).
3. Index dtypes as served; page tables over a shuffled pool. A sliding-window pool must sit
   behind a **non-identity** page map, and its page table must be a **different tensor** from the
   full one (assert both).
4. `writes(path, i)` / `kv_targets(m)` list exactly what launch i writes (a merged target launch
   writes the NEXT layer's cache). The harness poisons only those slots and counts every other
   changed byte as `kv_stray`.
5. Paths: `PATHS[0]` is the deployed path `call()` runs. Every contract oracle's `path` must be
   in `PATHS`, and `set_path` must select it.
   - split: target `runner.use_layer=False` (instance attribute, runner.py:183); draft STAGE=2
     (through the job env). Oracle legs run on the baseline overlay only.
   - native: the family contract freezes a `native` oracle, so `PATHS` must implement it unless
     the task drops it with a justified override (`scaffold_ut.py new --override`, visible in the
     frozen contract). Target = the per-stage torch references of sgl-amd-mono test/kernels
     combined over the ranks with `dist.all_reduce`; draft = `_Draft._attention_reference /
     _reference`. The baseline must pass the native floor; if it does not, fix native, never
     loosen the bound. Path names are free (the worked example calls the deployed one "merged");
     only the oracle `path` values of the contract must appear in PATHS.
6. `__init__(n_layers=None)` builds the call() step's layers; `n_layers=N` builds rank_chain's N
   distinct layers. The first layers of both must be identical (same seeds, same cache draw
   order): `Chain.check()` compares its step(1) with call()'s `layer_out[0]`.
7. Build the runner the way the serving builder does (Muse: `spec.py::_build`), and check every
   piece of state it sets that the constructor does not: e.g. the in-kernel wait deadline
   (`set_deadline`; left at 0 every wait fails at once) and pre-built sharded weights (Muse
   `runner.tpx_w` must be set before `bind`, otherwise `bind` builds them with SGLang's TP device
   group and an all-to-all the harness's gloo group does not have).
8. JIT builds: the first launch of a width compiles. A rank that compiles while its peers already
   spin in an in-kernel wait can trip the deadline; if the subject offers a compile-only mode or a
   warm-up, build the step's launches before the harness's barrier (step_meta / bind run before it).
9. Never import the subject package at module level of rank_model.py: rank_harness imports
   rank_model before `leg_env()` snapshots the process state, and the candidate package's
   import-time code must run after that snapshot. Import inside the methods.
10. `buffers()`: everything the launch writes that a skipped write would leave stale must be
   in `produced` (NaN'd before every launch) or be the `ids` buffer (-7). `inputs` are the
   buffers a launch READS from the previous launch (NaN'd before launch 0 only): a buffer that is
   both (Muse `attn_in`, which the split path's K1 reads) belongs in `inputs`, never `produced`,
   or the oracle path reads NaN. `shared_layouts()`:
   read the scratch / peer layouts from the subject's own modules, never hard-code offsets;
   what you cannot lay out goes in `kept` with the reason (it shows in
   `run/harness_report.jsonl`).

## 2b. A model that is not shaped like Muse (optional RankModel hooks)

`rank_harness.py` assumes Muse's shapes unless the model says otherwise. Each hook is optional;
absent, the behaviour is the one above.

- **The world** is the task's `world` const axis (the deployment's tp): `scaffold_ut.py` refuses a
  rank_chain task that does not pin it, and `RankModel.WORLD` is frozen from it (TP2 DeepSeek,
  TP8 Muse). Lease that many GPUs.
- **The stream** (`STREAM_SHAPE`, contract attr `stream_layout`): a stream row that is not
  `[HIDDEN]`. DeepSeek V4.1's mHC residual is `[rows, hc, hidden]` (`stream_layout=mhc`,
  `--axis hc=4`); the contract's stream / stream_in / stream_out / layer_out take that shape.
- **Side state of the stream** (`STREAM_AUX = (width, dtype)`): what travels with the stream
  between layers besides the next layer's input. DeepSeek's seam mixes post `[hc]` | comb
  `[hc, hc]` | pre `[hc]`, packed f32 (contract input `stream_aux`, outputs `stream_aux_in` /
  `stream_aux_out`, present for `mhc` only). `launch(i, rows, m, fb, stream, x, aux)` then
  returns `(stream, x, aux)`; a launch may return new tensors instead of updating in place
  (rank_chain's parity check takes the returned stream).
- **A KV record that is not a slot row** (`kv_bytes(layer, t, idx)` -> int64 `[rows, B]` flat byte
  offsets, `kv_decode(layer, raw)` -> float32 `[rows, KV_WIDTH]`): fp8_ds_mla pages hold their
  576-B data rows, then their 8-B scale words, so `index_fill_(0, slot)` and `.float()` would
  poison / decode the wrong bytes. The harness poisons exactly those bytes, excludes them from
  `kv_stray` and gathers them for `kv_rows`. Without `kv_decode` each byte is read as one e4m3
  code (the family's tolerance is one code). Cache tensors must then be contiguous.
- **A read-only cache** (DeepSeek's compressed cache, top-k rows): put it in `self.caches` past
  the written layers, in a pool of its own; the harness counts every changed byte of it as
  `kv_stray` at `[LAUNCHES - 1, pool]`.
- **A dotted package** (`PACKAGE = "vllm.models.deepseek_v41.amd.mono"`): the harness locates it
  without importing its parents (the candidate overlay's finder claims the dotted name).
- **No device deadline / error words**: `device_errors()` returns None and a stuck in-kernel wait
  is only caught by the harness's gloo timeout (DeepSeek); say so in `NOTES`.
- **No compile-only mode**: the first launch of a width compiles on every rank; without a device
  deadline the peers just spin meanwhile (DeepSeek). With a deadline, warm up before the barrier
  (rule 8).
- **No routing ids exposed**: `buffers()["ids"] = None`; the harness reports ids as not_produced.

## 3. Negative controls

`scaffold_ut.py new` writes one stub per category in the family's `required_control_categories`
(selected by `layer_kind`) — `kv_slot` three times (`min_controls_per_category`). For each:
`leaf` an editable leaf, `find` an exact, unique fragment, `replace` the bug, `expect` the
outputs it must break, `category` kept. Extra categories are allowed. The lint refuses a missing
category, a missing count, or a control without a category.

The lint also requires a negative control on EVERY editable leaf (meta.ut_contract.editable_leaves)
unless `leaf_control_waivers` names it with a why — so the required categories are a minimum,
not the whole list. Kinds of kv_slot bug worth covering: a wrong slot (offset / page), the right
slots in the wrong row order (only the per-row kv_rows check sees it; kv_stray stays 0), a
sliding layer bypassing the full-to-SWA map, a dropped K or V store.

The controls are judged against the baseline AND every oracle leg (`caught_by`,
`rejected_by` in the report): a control only the baseline comparison catches is a weak control.

A model whose controls are GPU-verified carries them as data, `references/mono_models.json`
`negative_controls`: the request carries them and scaffold_tasks.py writes them in place of the FILL stubs, so a new task of that
model starts from the verified set instead of re-deriving it. Rules every control must satisfy
(each was a real failure, DeepSeek-V4.1-Flash draft on 015, 2026-11-17):

- **Reached by every build the task runs.** A mutation inside a compile-time branch
  (`const_expr(...)`, a build flag such as a tuned / generic path) that some of the task's builds
  never compile is invisible there: the negative-control gate reports it "not detected" on every
  case of those builds. Put the mutation on the path the task's builds share, or give each build
  family its own control.
- **In range for every width and row layout.** An index mutation (a swapped / shifted row, slot or
  offset) must stay inside the step's tables for every served width, odd ones included: clamp it
  (e.g. the partner of the last row of an odd width is itself). An out-of-range store in a
  persistent kernel can corrupt the launch's own handshake words and hang every rank instead of
  failing: a control must make the outputs wrong, never stop the step.
- **Breaks numbers, not synchronization.** Drop or corrupt a value after it was received (keep
  every poll / wait / flag of the original); a control that skips a wait or a push turns into a
  deadlock on a kernel without a device deadline.

## 4. Manifest

- rows ⊆ the served widths; every observed width ≥ 1 timed, ≥ 1 random, every graph phase ≥ 2
  replay sigs sharing ONE row count (a graph is per width; the request picks the width with
  the most cases, its shortest and longest context; the lint refuses mixed widths). The other
  width is still exercised under graphs by the interleave sequence.
- r16 (or the larger width) needs ≥ 2 requests; neighbouring random cases have different seeds.
- run/ut_cases.json is the request's (scaffold_tasks.py installs the manifest); do not edit it.
  Calibration cases (e.g. context 1024 / 8192, to line the UT up with a layer bench) are optional
  extras the request may add.
- Two widths among the random cases enable the interleave sequence (graphs of both widths and
  eager steps back to back).

## 5. Timing

Timing is `rank_chain` only (`rank_harness.Chain`, through `cases.rank_chain`); `call()` is never
timed. `Chain.check()` runs the call() path once on the same case and fails if the chain times
something else.

Cost: a rank_chain leg pays ~18 s of start-up (leg_runner, 8 rank processes, first model build)
against 0.2-0.5 s of timed replays (Muse Spark target, gfx950). So one leg per side per rep times
EVERY undecided bucket on one set of rank processes (`rank_timing.time_buckets`, ~3 s a further
bucket), with the bucket order reshuffled and the first side alternated every rep; the early
exit stays per bucket (harness_lib `_measure_rank_chain`). Measured on the 23-bucket target UT:
one-bucket-per-leg timing was 43 min; batched medians match it within 0.5%.

For search iterations, `ut_driver --quick-timing` times only representative buckets (per phase
and m: the most-called one and the shortest and longest `axes.context`) and prints
`GEAK_QUICK_TIMING_SPEEDUP`. It is never a pass: accept a candidate only on a full run.

## 6. Done

`python3 ut_contract.py <task>` (lint) has 0 errors, then GPU correctness passes: the oracle
floor (baseline vs every oracle), replay + interleave, and every negative control caught.
Do not edit `rank_harness.py`, `ut_driver.py` or the family to get there; a GPU result that does
not match means the model (or the harness / template, reported upstream) is wrong.

Smoke test without the driver (8 GPUs): `python3 rank_harness.py <rows> <context>
[interleave <small ctx,..> <large ctx,..>]` runs every path, prints finiteness / repeatability /
deployed-vs-oracle per output and the device error words.
