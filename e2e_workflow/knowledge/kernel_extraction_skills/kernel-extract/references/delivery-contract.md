# Delivery and workflow acceptance

Read in `select`, `extract` and `extract_op`. The maintained implementation is
`SKILL_DIR/e2e_workflow.js`: `kernelSelectionVerified`, `hasFrozenBaseline` and
the extraction result schemas. `scripts/kernel_extraction_delivery.mjs` reads
these same definitions for CPU preflight. It does not execute GPU tests.

## Declare the boundary before extraction

Keep selection targets, the tested mathematical boundary and editable source
scope distinct. Required companions do not become additional selected targets.

| Requested boundary | Selection evidence | Task correctness and timing |
|---|---|---|
| One selected kernel, including an ordinary fused whole-op task | Anchor-kernel verdict; no `--required-device-kernel` | Complete declared callable; preserve its companions and check actual dispatch per case |
| Explicit mandatory chain, such as score → top-k or split-K → reduce | `KERNEL.joint_ut.phases[]` and a `joint_chain` verdict for every declared phase | Complete chain per invocation and phase; one task and standard UT entry |
| One whole op with alternative selected kernels | `KERNEL.dispatch_ut.phases[]` and separate anchor verdicts | Same complete callable for every branch, actual dispatch verified per case |

For `dispatch_ut`, each phase row has a unique branch label in `regime`, an exact
`device_kernel` and `seam_candidates`. Return corresponding rows through
`phase_selection_validations`; every row verifies the same whole-op callable.
Do not declare `required_device_kernels` for mutually exclusive branches or
combine `dispatch_ut` and `joint_ut` on one task. Retain the original workload
phase in the case manifest when branch labels subdivide PREFILL or DECODE.

For a joint task, **selection must put the declaration in the queued KERNEL**.
Each phase has a unique nonempty `regime`, an anchor `device_kernel`, at least
two distinct `required_device_kernels` including the anchor, and structured
`seam_candidates`. Keep the same phase labels in the return. Adding `joint_ut`
only to `meta.json` or the extraction return does not change the caller's KERNEL.
If the original declaration is wrong, report the mismatch for the selector to
repair; do not repeatedly capture or invent a different declaration to get PASS.
The workflow returns `needs_selection_repair=true` for a returned chain missing
from the original declaration. Repair the queued KERNEL from saved evidence
before invoking extraction again; retrying the same declaration cannot fix it.

“Deepest” is relative to the requested boundary. A score-only child cannot
replace a score/top-k chain. A fused MoE task must keep the whole-operation
`op_seam`: probe relevant outer/dispatcher/op-seam candidates for selection,
and test GEMM leaf execution separately in the UT. Do not pass single-GEMM
children as competing whole-op selection targets. Preserve the caller's
candidate classifications and probe every relevant allowed candidate.

Mandatory joint members must occur together in each verified invocation.
Mutually exclusive GEMM/top-k variants are not simultaneous mandatory members.
Keep a case manifest of actual phase/shape/dispatch branches and their reference
launches; verify alternative branches per case. For an explicitly requested
joint chain with different mandatory member sets, the caller must represent
those branches in its supported declaration and evidence before admission.
Do not silently drop variants, merge incompatible traces, rename a failing
joint verdict to a leaf verdict, or use a bare family name to hide a different
kernel identity. A missing representation is an explicit contract blocker.
Partition branch evidence using observed operands/state or a separately verified
dispatch marker, not by keeping only traces containing the desired device name.
Retain counts and evidence for the other branches. In-scope chain success does
not certify the whole workload or remove unprobed whole-operation candidates.
If source inspection finds additional whole-op entries, probe them and retain
the deepest compatible complete boundary; documenting their existence alone
does not satisfy the candidate-coverage gate.
When this changes `target_callable`, adapt cases to the actual callee signature
and state lifecycle, retaining the original captured values and conversion
provenance. Revalidate baseline/candidate binding, correctness, deployment mode
and timing for that entry. A passing selection probe cannot transfer the old
entry's full UT verdict to the new one.

## What must pass

1. **Real UT:** the standard `python3 unittest.py` exits successfully and runs
   the requested oracle/random correctness, graph/compile/input-refresh checks,
   actual candidate binding, negative controls and complete two-leg timing.
   Every positive-mass served signature has both latencies; retain missing-case
   mass and limitations. No speedup threshold or new model baseline is required
   for extraction acceptance.
2. **Editable source:** each kernel claimed as extracted has its actual native,
   JIT or generated implementation under `kernel_src/`, its build/import path
   and untimed execution proof. A copied dispatcher that calls installed leaves
   proves only dispatcher editability. Installed dependencies/fallbacks are
   allowed when explicitly outside the claimed editable scope; they cannot
   satisfy a request to extract those leaves. For a requested indexer chain,
   retain separate score and top-k native/JIT bad-source numerical rejection.
3. **Frozen baseline:** preserve the current serving stack and baseline hashes;
   prove baseline and candidate resolve through their separate environments.
   Return the applicable `baseline_frozen`, `candidate_bind` and baseline paths.
   The JS declaration check accepts `baseline_frozen=true`, a `candidate_bind`
   object, or a nonempty `baseline_callable`. This is a compatibility check,
   not independent proof that the baseline files are frozen or the legs differ.
4. **Selection evidence:** the original `kernel_selection.py` verdict must have
   `contract="kernel_selection"`, `ok=true`, matching exact callable and kernel,
   positive observed/marker/matched call counts, all relevant candidates tested,
   an admissible selected role, and verified boundary depth. Preserve failures.
   A joint return also needs exactly one `phase_selection_validations` row per
   declared phase, a matching mandatory-member set, positive evidence per
   member, and complete `joint_call_coverage`. The primary callable must be one
   of those verified phase entries.
5. **Serialized return:** obey the supplied role schema, including lowercase
   `smoke="pass"` for `extract_op` or `unittest_smoke="pass"` for `extract`.
   `task_dir` must identify the task and `editable` must not be false. String
   fields such as `notes`, `smoke_detail` and `reference_io_sha256` remain strings;
   put per-phase details in their structured fields or evidence files.

The workflow checks the return and selection evidence; it does not independently
rerun every GPU assertion or audit every native leaf. A passing JS gate is not
a substitute for items 1–3, nor for the requested workload coverage.

## Recorded source delivery

When the caller requests actual selected-kernel source, set
`KERNEL.source_delivery_required=true`. `source_device_kernels` may explicitly
add required companion leaves; the gate always requires all
`selected_device_kernels` (falling back to the anchor). A source request cannot
be satisfied by an op-only backend benchmark or by removing requested members.
Read [source delivery records](source-delivery-records.md) for the manifest
and record format. The production workflow and delivery preflight both check
these records against current source, inputs, harness and target_callable.

## Preserve machine evidence without model transcription

Small verdicts may be returned inline as `selection_validation`. Keep all gate
fields intact, including `joint_call_coverage` as an array. `_same_as`, prose
paths, `_elided_for_transport` summaries and a lone `ok:true` are not supported
references. Do not manually rebuild a verdict from the desired outcome.

Prefer `selection_validation_ref={"path":"run/selection.json","sha256":"..."}`
for large verdicts. The path is relative to `task_dir` (absolute paths inside
the same task also work); the digest is SHA-256 of the exact file bytes. The
workflow verifies the path/hash, reads that complete JSON and applies the same
gate. A missing, changed or invalid file fails even if an inline PASS is present.
The file must remain available through workflow acceptance and later handoff.

For joint returns, each `phase_selection_validations[]` row contains `regime`,
`target_callable`, `seam_candidates`, and its own inline verdict or file/hash
reference. Repeat the primary phase reference at top level; do not replace its
phase verdict with an informal reference to the top-level field.

Save the exact original `KERNEL` input as `run/requested_kernel.json` and author
the schema-correct result skeleton as `run/draft_return.json`. After GPU tests,
attach raw verdicts and preflight without re-running capture or timing:

```bash
node "$SKILL_DIR/scripts/kernel_extraction_delivery.mjs" \
  --kernel "$TASK/run/requested_kernel.json" \
  --result "$TASK/run/draft_return.json" --phase extract_op \
  --verdict run/selection_DECODE.json \
  --phase-verdict DECODE=run/selection_DECODE.json \
  --phase-verdict PREFILL=run/selection_PREFILL.json \
  --out "$TASK/run/RETURN.json"
```

The draft must already contain the two phase rows; use the actual phase labels
and verdict paths. For a non-joint task, omit both `--phase-verdict` options and
choose `--phase extract` or `extract_op` to match the invocation. The helper
computes hashes, replaces only the designated inline verdicts with references,
and writes `--out` only when the exact serialized delivery passes. Return that
JSON unchanged. It never changes the caller's declaration or any verdict's
success/failure. To check an already packed result, omit the attachment options.

On failure, read `checks.schema` and `checks.selection` first. Repair schema,
reference or missing-field errors from the saved raw evidence. Re-run GPU work
only for changed code/inputs or an unresolved execution/correctness/coverage
defect. A real missing member or failed selection verdict remains a failure;
return it with evidence instead of retrying an unchanged UT until the budget
expires. Report UT pass, delivery acceptance, coverage and E2E results separately.
