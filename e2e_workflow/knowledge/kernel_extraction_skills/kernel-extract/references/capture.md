# Bounded live capture without a workflow runner

This is a fallback during extraction, not a step inside the UT. Prefer seeded
inputs from the fixed case manifest in [task authoring](task-authoring.md#seeded-inputs-by-default).
Use existing shape/dispatch metadata first; capture only evidence or structural
state that cannot otherwise be recovered. Numerical test payloads do not require
serving capture. Runtime launch attribution remains a separate selection check.

Read the user's actual serving launch and request-client commands. Reuse a
compatible trace/capture before launching another service. If a probe is needed,
write a task-local capture launcher adapted to that environment. There is no
universal server-start command in this skill: flags, model configuration,
health API, tokenizer and client APIs must come from the installed backend.

Record request count, concurrency, token/context controls, timeout, capture
window and case/total byte budgets before launch. Inspect wrappers and warmup
hooks: zero measured repeats need not mean zero client requests. Disable any
outer full-replay warmup for this bounded probe and verify the effective client.
Probe measurements must be labeled diagnostic and cannot replace baseline
throughput or the original workload distribution.

Keep serving TP and allocated devices unchanged. Run an owned serving probe
under a full-set lease, with the launcher in the foreground for the entire
server/client lifecycle. It may start child processes only if it records their
PIDs/process group, traps exit/signals, stops and waits for those owned children
before returning. Apply bounded health polling; stop polling if the server
exits. Do not attach cleanup to unrelated processes or stop an existing service
unless the user authorized that lifecycle. Isolate a temporary capture overlay
from the accepted baseline and restore it after the probe.

For Python callables use the shared overlay and trace hooks; inspect CLI help
for exact options. A typical preparation sequence is:

```bash
python3 "$SHARED_TOOLS/overlay_setup.py" add-capture \
  --target "$TARGET_CALLABLE" --out "$CAPTURE_DIR" \
  --overlay "$CAPTURE_OVERLAY" --from "$BASELINE_OVERLAY" \
  --capture-file "$SHARED_TOOLS/capture_shapes.py"
python3 "$SHARED_TOOLS/overlay_setup.py" add-marker \
  --target "$TARGET_CALLABLE" --overlay "$CAPTURE_OVERLAY" \
  --marker-file "$SHARED_TOOLS/seam_trace.py"
```

Pass the capture overlay through the serving process's `PYTHONPATH`; preserve
the existing baseline module/search paths and environment. Mark all relevant
candidate callsites, including cached aliases, with separate `add-marker`
calls as appropriate. Native/JIT kernel objects may not be safely monkeypatched;
trace their real Python launchers and native launches instead.

Capture defaults to `CAPTURE_PERSIST_POLICY=metadata_only` in both the library and
GEAK. It saves JSON metadata, complete shape/layout/scalar signatures and call counts;
it neither snapshots tensors nor writes `reference_io.pt`. `max_cases` does not cap
this histogram. `reference_io=null`, `oracle_status=not_captured` and `num_tensor_records=0`
are normal; validate `shape_counts` and launch evidence instead of requiring a golden file.

If structural state cannot be reconstructed, explicitly select `structure_only`
and name additional structural operands with `CAPTURE_STRUCTURAL_OPERANDS`. Defaults
are 64 MiB per process and 8 MiB per case, enforced before CPU copies; skipped data
still contributes shape metadata. Packed integer weights require format-aware
constructors, not integer-to-float random casts. Legacy `share_large`, `moe_slim`
and `full` policies remain explicit opt-ins. See [task authoring](task-authoring.md#capture-policy).

Read the helpers' environment controls before choosing budgets and profiler
windows. Do not assume their defaults capture every relevant shape or phase.
`capture_shapes.py` handles tensors and ordinary containers; opaque `self`, KV
getters, routing/state or multiple phase targets may require a task-local
serializer. Follow [source extraction](source-extraction.md) for byte accounting,
shared-state, independent phase state and replay checks. Keep customizations
task-local, with source hashes and a bounded small replay before a larger sweep.

Once captured, verify target-call markers, host-launch correlation, real device
kernel names, grids, invocation counts, phase identity and rank integrity using
`kernel_selection.py --strict-launch`. Pass `--required-device-kernel` only for simultaneous
mandatory members, not alternative dispatch. Run `--help` for its input contract.
Store its complete JSON, raw trace and metadata. The CLI may promote/reclaim
oracles under `--task-dir`; use a separate directory per phase and preserve
needed captures before allowing that cleanup. No evidence from one phase may
overwrite another phase's oracle.

When capture was necessary, replay one baseline case per declared phase before expanding source
extraction. Successful file persistence alone does not prove complete inputs,
correct state reconstruction or faithful whole-call timing.
