# Shared sampling before multi-config extraction

Use this path for every workload: one `--config` for a single config, or repeat it
for several configs. Read
each config's scheduler observations first; an endpoint-only trace does not establish
the middle configs' shapes, frequencies or dispatch branches. Observations can come
from existing baseline/CI logs. Collect a short observation window only where the
required raw rows are missing; do not repeat a full workload just to plan points.

## Inputs and executable policy

Each input directory contains `serving.json` and `evidence/observed_workload.json`,
using the ordinary probe/observed-workload schemas. Keep source files immutable during
the campaign. Supply a common `--runtime-identity identity.json`, or a separate
`runtime_identity.json` in each config directory. It records:

```json
{
  "image": "the deployed image digest",
  "model_revision": "the deployed model snapshot or content manifest hash",
  "source_revision": "installed source/overlay revision or content manifest hash",
  "gpu_arch": "the actual GPU architecture"
}
```

Additional identity fields (e.g. driver/runtime versions or GPU SKU) participate in
equality. Equal names without matching revisions are insufficient. Record actual
values from the environment receipt, not placeholders from this example.

The pipeline scenario can specify:

```json
{
  "multi_config": true,
  "selection_top_k": 10,
  "selection_overall_pct_gt": 3,
  "selection_phases": ["PREFILL", "DECODE"],
  "sampling": {
    "mode": "shared_union",
    "num_points": 24,
    "context_tolerance": 1.5,
    "target_pct": 90
  }
}
```

The profile stage is agent-shaped: pass the numeric `sampling` fields to the command
below. UT generation uses `ut_wave.py start --timing none`; performance measurements
belong to optimization. `num_points` is
one global initial budget, **not** points per config and not a fixed skill constant;
24 above is an example. Choose it from the number of distinct deployment/phase domains
and the available capture budget, then inspect the reported gaps. The >3% cutoff is
per complete operator in **any config's own whole-workload GPU-time denominator**, with
phase Top-K applied first. Keep required companion kernels/phase branches below 3%.

## Plan, capture, project

Set `PROFILE_SLOT=$WORKLOAD_ROOT/stages/profile`. In a new output directory:

```bash
python3 "$PROFILE_SLOT/multi_config.py" plan --group-dir "$OUT" \
  --config c2="$OUT/configs/c2" --config c4="$OUT/configs/c4" \
  --runtime-identity "$OUT/runtime_identity.json" \
  --num-points 24 --context-tolerance 1.5 --target-pct 90
python3 "$PROFILE_SLOT/multi_config.py" capture --group-dir "$OUT" --dry-run
```

Planning is CPU-only. `sampling_plan.json` records candidates before/after the union,
the selected capture list, per-config row/count attribution, deployment groups and
unselected mass. `sampling/<deployment-id>/` holds each shared pool's inputs.

Compatibility includes all serving fields/arguments/environment except transport
addresses/ports, physical GPU allocation, top-level request concurrency and `_` notes.
Scheduler settings, graph limits, prefill chunk, KV policy, TP/EP and unknown options
remain significant. In particular `--max-running-requests` is not stripped as if it
were a client concurrency label. Different deployments get separate servers/pools.

Within a pool, deduplication uses phase, exact shape, sequence count, observed graph
mode and a bounded context cell. It does not assume adjacent M values use the same
kernel. The context representative is an actual observed row; context clustering is
an explicit approximation, not proof that attention/page/routing state is identical.
Use `--context-tolerance 1` for exact context equality. Operator-specific standalone
UTs may later have a smaller valid reuse signature (e.g. a dense GEMM needs no KV axis).

The budget is allocated greedily to uncovered, normalized per-config/per-phase proxy
mass. Log duration cannot make one config dominate. Middle-only domains remain
candidates. `budget_limited` and each phase's unmet target are explicit; never label
an unmeasured phase zero cost or the planned proxy a GPU-time coverage percentage.

Once the planned deployment pools can run under the established GPU leases:

```bash
python3 "$PROFILE_SLOT/multi_config.py" capture --group-dir "$OUT"
python3 "$PROFILE_SLOT/multi_config.py" project --group-dir "$OUT" --merge
python3 "$PROFILE_SLOT/multi_config.py" select --group-dir "$OUT" --params "$PARAMS"
```

`capture` holds each pool's full GPU set through `shared/gpu_lock_set.py`, then starts
one server using `probe_server.py`. It runs `run_points.py --all-evidence` (profile evidence only): a kernel small in the pooled discovery
log must not lose its source evidence before the per-config >3% test. GPU commands run
in the deployed image with the configured devices; `--dry-run` only prints argv arrays.
Pools run sequentially by this helper; users with separately leased device sets can run
the printed commands concurrently. Do not put two pool servers on the same devices.

`project` reuses the original census/probe/shape evidence via references and writes
`projected/<config>/`. It recomputes each config's represented instances and ranking
from **that config's own log**, then `--merge` calls the existing multiconfig merger.
Raw measurements retain the capture pool and signature; a shared capture is never
presented as an independent serving run of every config. No task or frozen UT is
rewritten by this step. UT cases retain their config IDs and original weighting rules.

Projection rechecks achieved shapes/context and sequence count. Outside-tolerance or
unsampled discrete domains remain uncovered bands, with their nearest-point cost
estimate explicitly remaining a model. A wholly unmeasured phase stops ranking for
that config. Inspect `sampling_projection.json` and normal coverage artifacts before
claiming achieved coverage. Collectives and other unselected work remain in the
denominator; shared standalone evidence does not establish whole-serving coverage.

## Resume and fill gaps

With unchanged inputs, increase `--num-points` and rerun `plan`: point IDs and the
greedy prefix are stable. `capture` reuses signed censuses and captures only missing
points; `project --merge` then reweights the expanded pool. A failed or shape-shifted
probe needs the ordinary probe diagnosis/recapture as well; increasing a planning
budget is not proof that it achieved the requested shape. Changed input hashes or
deployment identities require replanning. Changed capture signatures require explicit
recapture or a new pool, never reuse of an unsigned census.

The context binning and strict deployment matching favor conservative reuse. The
planner can therefore report little or no cross-config sharing when a CI sweep changes
the server configuration along with concurrency. Report that result rather than
loosening compatibility to manufacture a saving.

## UT parallelism

UT authors run in CPU-only task lanes (`--author-workers`, default 8). Default
`--gpu-mode on-demand` routes each GPU command through `ut_wave.py gpu -- ...`, leasing
only its required device count from the campaign pool and releasing it after workers
drain. Multi-GPU requests wait without holding partial sets. Do not hold an outer pool
lease around this wave; its host broker shares the existing GPU lock files. The default
`ut_wave.py start --timing none` finishes with `unittest.py --validation-mode correctness
--reuse`; it runs no timing tail. Preserve all weighted shapes for optimization and
report `performance_measured: false`.

For an explicitly requested performance wave, `--timing tail --timing-workers N`
times tasks on disjoint GPU sets after authoring drains. Each B/C pair stays on one
device, multi-rank tasks retain an exclusive tail, and measurements retain their cache
and noise policy. This optional mode does not change the extraction default.

Keep numerical gates and draws unchanged. Use `--quick` for debugging, then correctness
qualification; reuse compatible passing gates. Large inline tensor bundles and full-suite
temporary oracle storage can still consume time independently of timing.
