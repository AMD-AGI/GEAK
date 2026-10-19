# Local delivery contract

Save the task declaration, perform extraction and record evidence. The shared
`kernel_extraction_delivery.mjs` validates the result against the workflow's
schemas and source evidence. This checks evidence consistency, not GPU execution.

## Declare the task before extraction

Save the original request as `run/requested_kernel.json`. Its JSON object is
called `KERNEL` by the local checker; no external queue or orchestrator is involved.

```json
{
  "short_name": "example_operation",
  "entity_kind": "gpu_kernel",
  "op_kind": "attn",
  "device_kernel": "exact_selected_device_symbol",
  "selected_device_kernels": ["exact_selected_device_symbol"],
  "source_delivery_required": true,
  "seam_candidates": [{
    "target_callable": "package.module:complete_operation",
    "depth": 1,
    "role": "op_seam",
    "device_kernels": ["exact_selected_device_symbol"]
  }]
}
```

The example names are placeholders, not runnable kernel evidence. Derive actual
candidate entries and depth from installed source and runtime traces. Candidate
roles are `outer_wrapper`, `dispatcher`, `op_seam`, `inner_launcher` or
`kernel_entry`; a native/JIT object may be source evidence without being a safe
Python monkeypatch target. Include cost/workload links from the task plan.

Keep selected targets, the complete tested math boundary and editable source
scope distinct. `source_device_kernels` may add required companion leaves;
the source checker always retains every `selected_device_kernels` identity.

| Boundary | Declaration and verification |
|---|---|
| One anchor, including a whole fused op | Exact anchor verdict; include companions in correctness/timing and per-case dispatch |
| Mandatory chain, such as score→top-k or split-K→reduce | `joint_ut.phases[]`, with a `joint_chain` verdict for every declared phase |
| Alternative selected branches of one whole op | `dispatch_ut.phases[]`, with a separate anchor verdict for each branch at the same complete callable |

Each `joint_ut.phases[]` row contains unique nonempty `regime`, anchor
`device_kernel`, at least two distinct `required_device_kernels` including the
anchor, and structured `seam_candidates`. Every required member must occur in
each verified invocation. Distinct branches with different mandatory members
need their own labels and evidence. Return matching `phase_selection_validations`.

Each `dispatch_ut.phases[]` row contains unique branch `regime`, exact
`device_kernel` and `seam_candidates`. All branch verdicts verify the same whole
operation. Retain actual PREFILL/DECODE phase separately in case metadata when
branch labels subdivide a phase. Do not combine `joint_ut` and `dispatch_ut`.
Do not list mutually exclusive branches as simultaneous mandatory members.

“Deepest” is relative to the complete requested boundary. A score-only child
cannot replace a joint indexer; a GEMM child cannot replace a whole fused MoE.
Probe all relevant candidates for that boundary and retain actual leaf binding
checks separately. Partition traces using independently observed operands/state
or dispatch markers, not by discarding calls missing the desired kernel.

If the saved declaration is wrong, revise the task plan using evidence before
continuing. Do not infer a different request from a passing result. A changed
entry requires signature/state adaptation and new binding, correctness, graph
and timing checks; a selection probe cannot inherit an old entry's UT pass.

## Required real tests

- Complete executable UT: oracle/random-value correctness, full output semantics,
  deployment modes and input refresh, actual leaf execution, numerical negative
  controls, restored-source checks and both timing legs for all weighted cases.
- Actual native/JIT computation source for all requested leaves, build/import
  instructions and resources. An installed-library dispatcher is insufficient.
- Frozen baseline identity and candidate isolation in separate processes.
  `baseline_frozen:true` is only a declaration; retain the actual hashes/bind checks.
- Complete raw selection verdicts with positive observed/marker/matched calls,
  exact entry/kernel, all relevant candidates tested and valid boundary depth.
  Joint phases require matching members and complete per-call chain coverage.
- Recorded source/test snapshot and raw hashed GPU logs as defined in
  [source records](source-delivery-records.md). Missing coverage remains explicit.

## Result forms and preflight

Use one of the two local forms; both require source evidence here:

| `--phase` | Required status fields |
|---|---|
| `extract` | `editable:true`, `task_dir`, `unittest_smoke:"pass"` |
| `extract_op` | `op_kind`, `task_dir`, `smoke:"pass"`; also set `editable:true` |

Use `op_kind=gemm|attn|moe` for the corresponding head operators. The workflow
accepts `attention` as an alias for `attn`; declaration and result must agree.

Both forms also supply `target_callable`, applicable `candidate_bind` and
`baseline_frozen`, exact `device_kernel`, `seam_candidates`, full selection
verdict or hashed reference, and `source_delivery_ref`. Keep `notes`,
`smoke_detail` and `reference_io_sha256` strings. Use `"fail"` for incomplete or
failed tests. The complete optional-field schemas are in `e2e_workflow.js`.

Prefer `selection_validation_ref={"path":"run/selection.json","sha256":"..."}`
over transcribing large machine verdicts. The referenced JSON must be a complete
raw `kernel_selection.py` verdict inside the task. A changed/missing file fails;
an inline PASS cannot override it. Aliases such as `_same_as` or elided gate
fields are not supported. Keep these files through delivery and later review.

For a joint/dispatch result, each `phase_selection_validations[]` row has
`regime`, `target_callable`, `seam_candidates` and its own full verdict/reference.
Repeat the primary phase reference at top level. Save the draft result and run:

```bash
node "$SHARED_TOOLS/kernel_extraction_delivery.mjs" \
  --kernel "$TASK_DIR/run/requested_kernel.json" \
  --result "$TASK_DIR/run/draft_return.json" --phase extract_op \
  --verdict run/selection_DECODE.json \
  --phase-verdict DECODE=run/selection_DECODE.json \
  --phase-verdict PREFILL=run/selection_PREFILL.json \
  --out "$TASK_DIR/run/RETURN.json"
```

The draft must already contain those phase rows. Use actual labels and paths;
omit phase options for a single-anchor task. The helper attaches references,
checks hashes and writes `--out` only on pass; it never changes a raw verdict.
An already packed return can be checked without attachment options.

On failure, inspect schema, selection and source checks. Repair return-only
errors from saved evidence without rerunning a passing GPU UT. A real execution,
source or coverage defect requires fixing and retesting that defect. Report
GPU status, delivery consistency, coverage and whole task-set completion
separately. No speedup threshold is needed for unmodified source extraction.
