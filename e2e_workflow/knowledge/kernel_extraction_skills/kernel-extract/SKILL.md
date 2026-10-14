---
name: kernel-extract
description: Select top GPU kernels, extract editable source and build workload-weighted standalone UTs for AgentX trace replay across models and backends. Excludes fixed ISL/OSL and non-AgentX runs.
---

# AgentX kernel extraction

Applies to AgentX `kernel_extractor` phases `select`, `extract`, `extract_op`,
and their routing/bakeoff consumers, refreshes and resumes. Activation is
independent of roofline and expert-skill toggles. Preserve the configured
benchmark metric, measurement lifecycle, serving TP/GPU set and frozen baseline.
Keep probes, task packages and evidence under `EVAL_DIR`.

## Select from the completed baseline

`PHASE=select` replaces separate Profile and Strategize passes:

1. Read `TASK`, supplied UT references, `BASELINE_LOG`, `BASELINE_BENCH_DIR`,
   `BASELINE_SUMMARY` and effective model/serving configuration. Identify the
   completed measured leg and exclude warmup/discarded legs from its census.
2. Use logs to establish phase/shape distributions and sampling weights; use
   runtime traces to establish actual kernel identity and cost. Request ISL/OSL
   summaries do not determine kernel M or executed tokens after caching,
   batching and speculation. Discover applicable features from this stack.
3. Reuse compatible evidence. Fill missing costs, dispatch or operands with
   bounded targeted captures; do not launch another full AgentX replay or outer
   warmup for discovery. Follow [workload measurement](references/dynamic-workloads.md).
4. Apply [selection scope and coverage](references/selection-coverage.md), including
   its deterministic selector. The default is PREFILL Top 5 or DECODE Top 5,
   filtered by overall device-time share strictly >3%, with no backfill.
   Preserve the complete inventory, prior-task reconciliation and blockers;
   execution budgets limit scheduling, not discovery.
5. Declare one owner per complete UT boundary using the
   [delivery contract](references/delivery-contract.md). Preserve required
   companions and each selected identity without double-counting joint/leaf work.

Write `kernel_workload.json` and `kernel_ranking.json` under `EVIDENCE_DIR`.
Return their paths as `profile_workload_json` and `kernel_ranking_json`, plus
`head_candidates`, `kernel_candidates`, `config_directions`, `drop_list` and
`strategy_path`. Candidates retain the existing architect schema, exact
`device_kernel`, `entity_kind`, seam evidence and `kernel_extraction` cost links.
Return every evidenced selected target without truncating to the task budget;
unevidenced targets stay in the inventory with a blocker. Do not author UTs
in `select` or fabricate trace-window profile files from logs.

For refreshes, compare current accepted-stack logs with saved evidence and
invalidate only affected costs, identities and weights. On resume, reuse
compatible evidence and queues without repeating the baseline.

## Extract and validate the selected tasks

Read [source extraction](references/source-extraction.md) and the
[delivery contract](references/delivery-contract.md) before authoring or returning
an `extract`/`extract_op` result. They define the executable task, source and
binding proof, numerical checks, evidence transport and CPU delivery preflight.
A copied wrapper calling installed leaves does not establish source extraction.
A CPU gate cannot substitute for real GPU execution and correctness evidence.

Conditional task guidance:

- **Indexer score → top-k:** [unified indexer UT](references/joint-indexer-ut.md).
  Preserve one task covering served prefill/decode phases. Honor `AUTHORING_STAGE`
  when present; bootstrap replay readiness is not full UT acceptance.
- **Collectives:** [multi-rank UT](references/collective-ut.md). Preserve rank count
  and native communicator state, and validate every rank.
- **Requested source delivery:** [machine records](references/delivery-contract.md#source-delivery-records).
  Preserve requested selected/companion leaves and `source_delivery_required`.

Authoring can proceed in batches. GPU tests may overlap only on exclusively
leased disjoint physical devices. Run from each task directory:

```bash
cd "$TASK"
bash "$SKILL_DIR/../kernel_workflow/scripts/gpu_lock.sh" "$GPU_ID" python3 unittest.py
# Multi-rank task: lease the whole allocated set.
python3 "$SKILL_DIR/scripts/gpu_lock_set.py" "$UT_GPU_SET" -- python3 unittest.py
```

A comma list to `gpu_lock.sh` leases one device from a pool; `gpu_lock_set.py`
leases every requested device. Preserve container/allocation mappings and keep
GPU children within the lease lifetime. Serving captures use the unchanged
`bench_e2e.sh` lifecycle and full `SERVING_GPU`/`SERVING_TP`; reserve that whole
set against UT jobs, or serialize captures and validation.

## Observed workload score

Call `h.serving_weighted_speedup(per_case, meta)` with
`workload.kind="agentx_trace_replay"`, `workload.weighting="observed"`, and
`workload.serving_weight_model` containing `source="observed_workload"` and an
`evidence_path` to the workload artifact. Each `workload.cases[]` row contains:

| Field | Requirement |
|---|---|
| `sig` | Unique phase/shape/state/dispatch signature shared with timing results |
| `regime` | Observed phase, matching results and `served_regimes` when set |
| `effective_calls` | Finite nonnegative call mass on a common basis; may be fractional |
| `m` | Actual GEMM M when applicable |

Score `sum(baseline_ms * calls) / sum(candidate_ms * calls)`. Derive weights
from the original joint distribution and measured costs or validated interpolation;
record any conversion from time mass to calls. Keep unchanged cases in the
score. Every positive-mass served case needs both measured latencies; missing
models/cases, duplicate signatures and invalid costs fail instead of falling
back to synthetic weights or renormalizing survivors. Zero-mass probes provide
correctness evidence only. Existing `agentx_log_fold` tasks remain supported.

Bakeoff and author lanes preserve these cases, weights and binding checks.
`served_buckets` can narrow preliminary GEMM probes, but only the complete UT
validates repeated M with different state and non-GEMM axes. Report inventory
reconciliation, requested-task completion, tested-case coverage and e2e results
separately; per-kernel speedup or a passing subset does not certify the full workload.
