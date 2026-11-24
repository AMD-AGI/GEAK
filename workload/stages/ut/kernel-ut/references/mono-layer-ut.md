# Writing a mono_layer UT (multi-rank monokernel layer)

A `mono_layer` task tests ONE layer launch of a persistent whole-layer monokernel
(sgl-amd-mono: Muse Spark target / draft, ...) on every TP/EP rank, with real in-kernel peer
exchanges. `scaffold_ut.py new --family mono_layer ...` writes everything except the model:

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

Copying their structure is expected; copying their numbers is not — every constant must come from
your model's config and server evidence. Which oracles a model's example covers is
`references/mono_models.json` `oracles_ready`; a family oracle no example covers is dropped from the
task's contract by the request (`scaffold.contract_overrides`, with a why), never left for you to invent.

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
  `oracles_ready` dropped from `oracles` (and a `reference` that named it moved to a kept oracle),
  each with the reason; `scaffold_tasks.py --check` refuses an override without a why;
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
