# GLM MQA from ORIGINAL

This task starts with the unmodified GLM5.2 MQA callable. `prepare.py` copies
five pinned ORIGINAL/reference/evaluator files and initializes the editable
module from the reference bytes. It never reads the external `kernel_src/`,
past results, histories, transcripts, or optimized code. The external task
sources are not bundled in this repository.

```sh
PYTHONPATH=. python3 examples/tasks/glm_mqa/prepare.py \
  --source /path/to/glm52_dsa_gluon_mqa_logits_task \
  --out /new/task/directory \
  --image registry/image@sha256:MANIFEST_DIGEST \
  --config-digest sha256:OBSERVED_IMAGE_CONFIG_DIGEST
```

The output has `task.json`, `objective.json`, the editable module, and the
protected files. The runtime owner must verify the declared image/config and
device before installing the task. Preparation starts no process or GPU work.

## Editable and protected files

| File | Rule |
| --- | --- |
| `kernel_src/fp8_mqa_logits_seam.py` | Edit the complete module and callable implementation. |
| `baseline_src/fp8_mqa_logits_seam.py` | Frozen ORIGINAL reference. |
| `ut_main.py`, `harness_lib.py`, `weight_lib.py`, `meta.json` | Frozen evaluator, numerical/state contract, and timing method. |

There is no C++ helper in ORIGINAL MQA. Additional compiled helpers require a
declared source/build closure before use. The same evaluator method can read
BMM results, but BMM needs its own ORIGINAL-only task preparation. Its ORIGINAL
source is Python; copied optimized C++ must not enter the engineer packet.
An undeclared installed optimized module is not a candidate source.

The exported entry is:

```python
glm52_fp8_mqa_logits(Q, KV, kv_scales, weights, cu_starts, cu_ends, clean_logits)
```

`clean_logits` is required at this seam; the workload passes `False` by keyword.
Q has shape `[M, 32, 128]`, KV `[K, 128]`, both use the architecture's FP8
flavor. Scales `[K]` and weights `[M, 32]` are FP32. Starts/ends `[M]` are
INT32. All six tensors are inputs. The callable returns a fresh FP32 `[M, K]`
logits tensor; output allocation stays inside full-call timing. The covered
starts are zero, and ends are the ramp `K-M+1` through `K`.

Correctness is the unchanged UT's bitwise comparison on the written region,
plus architecture, binding, dispatch, shape, graph diagnostic, and KV
mutation/restore checks. Unwritten cells under `clean_logits=False` have no
promised values. This is differential validation against ORIGINAL, not an
independent mathematical oracle or qualification of all shapes/layouts.

| Case | M | K | Phase calls | Timing |
| --- | ---: | ---: | ---: | --- |
| prefill_extend_m64 | 64 | 573440 | 21000 | Eager CUDA events |
| prefill_extend_m1856 | 1856 | 116480 | 21000 | Eager CUDA events |
| prefill_draft_m64 | 64 | 573440 | 1000 | Eager CUDA events |
| prefill_draft_m1856 | 1856 | 116480 | 1000 | Eager CUDA events |

These cases cover 2.4336% of the historical operation-compute census. M11072
and M32768 remain unqualified. Extend and draft remain separate folds.

The unchanged dispatch check requires the expected Gluon kernel name in both
arms and the M-derived candidate grid axis. Architecture checks also require
the Gluon exports, and binding checks reject `torch.ops` registration and the
compile guard. Many module/dataflow changes remain possible, but an arbitrary
backend or kernel-name replacement does not yet satisfy this harness. Do not
defeat those checks to claim success; a wider backend contract needs separate
prospective qualification.

## Measurement and handoff

The existing Arena evaluator still owns request IDs, source/epoch binding,
launch/settlement callbacks, raw artifacts and positive cleanup. The external
runner executes unchanged `ut_main.py` in three fresh processes inside that
owned operation. Each process uses the existing five legs, discards the first
leg, then takes the minimum of the remaining legs; warmup=5, repeats=20,
inner=1. Baseline/candidate/control timing and per-case null floors are retained.

The native gate needs at least **two** distinct valid processes before it can
assess repeatability. This replay policy requires **three** and keeps all three
results. The final gain must exceed each process's measured null floor in
**all three** processes. A case vetoes the gain if its regression exceeds its
null floor in **all three** processes. Setting
`GLM_UT_REQUIRE_REPRODUCIBLE_GAIN=0` avoids treating the first missing history as
a correctness failure; it does not remove the final native gain gate.

The runner forwards an explicit list of runtime, device, cache and compiler
environment settings and records exactly those values in `START.json`. Unset
settings remain unset; credentials and unrelated workflow settings are omitted.
UT reports are retained through regular-file reads that reject symlinks. Each
process receipt includes hashes for its reports, history, stdout and stderr.

The `external_primary` objective is the median of the three native
`prefill_extend` folds. Each process fold is
`sum(calls * baseline_ms) / sum(calls * candidate_ms)`. Export also requires the
native reproducible-gain gate and no reproducible per-case regression. Every
phase and case remains in the result. Per-case median times are descriptive;
they are not substituted into the headline formula. Eager event timing can
include host submission gaps. These results do not establish serving gains or
roofline percentages.

The owner runs the contract's single `evaluate` command in its existing
evaluation container, then supplies the reader's observations and retained
artifacts to the ordinary Arena settlement. The wrapper does not create
cleanup evidence or allocate devices. Both GEAK and GEAK-EVO must use this
same task/method/objective for a common-input comparison. The original control
launcher's scoring adapter must use the native primary field as well; a generic
geomean parser would change the experiment.

For an Astra pilot, use the same pinned model profile, task, evidence exposure,
runtime and independently recorded 2–3 hour budget in both arms. The copied
optimized endpoints belong only to fixed replay. Keep their source, prior
results and explanatory narratives outside the engineer/planner input tree.
Save new source, parent/idea IDs, measurements and concrete evidence with the
existing archive APIs; do not add a parallel history database.

Declare the release's always-on `trial_history` in the GEAK-EVO FULL treatment:
it changes planner/engineer inputs even when other optional mechanisms are
disabled. For the first qualification keep both per-evaluation
`report_context` and the JS `report_update_instruction` disabled. Enabling only
one is not a coupled reporting treatment; enabling both later requires their
existing adapter settings to agree and be recorded. The archive's existing
development ranking remains a descriptive search heuristic; endpoint export
uses the native primary score and gain gate described above.

`meta.files.ut_main.py` contains an older digest. The stager pins the actual
unchanged UT bytes (`a51d6e50…`) and preserves that discrepancy; it does not edit
metadata to hide it. Supplementary external `evidence/` files are not staged
or required by UT execution. Hardware-evidence exposure for the pilot must be
reviewed and declared separately in both arms.

The source and CPU adapter checks do not establish installed owner support,
a completed ORIGINAL GPU evaluation, search engagement, or a finished pilot.

The remaining runtime work belongs in the existing provider, not the engine:

1. `native_host_adapter.evaluator_for_driver` already supplies
   `NativeCallbacks.launch/settle` to `ArenaEvaluator`. Its current helper only
   supports its installed native/generated runners. Add this external method
   to that helper's existing task preparation and observer dispatch, using a
   prospective admitted closure. Preserve the running helper installation.
2. Materialize the exact request and protected task plus candidate source into
   that operation's private inputs. Expose the pinned GEAK package read-only
   inside the declared `ac30…` image, so `geak_evo.evaluate.external_runner` is
   importable. Supply absolute request/task/output paths; the relative argv in
   the task is the command template, not an assertion that those mounts exist.
3. Retain all wrapper artifacts under the owner's `observer/` namespace. Use
   `read_external_report(request, artifacts)` for the observation groups, and
   include its evidence paths with that prefix in the ordinary host receipt.
   Only the owner records actual runtime, positive cleanup, costs, and timeout
   settlement. A wrapper return code alone cannot supply those fields.
4. Run the complete ORIGINAL task once through that path (it performs three UT
   processes), then use the same path and `objective.json` for GEAK and GEAK-EVO.
   Verify the control launcher consumes the checked native primary score and
   gain flag. The engine's command/dispatcher support is source-tested; the
   external provider installation and control-launcher connection remain open.
