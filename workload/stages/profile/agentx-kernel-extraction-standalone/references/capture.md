# Bounded live capture without a workflow runner

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

For Python callables use the bundled overlay and trace hooks; inspect CLI help
for exact options. A typical preparation sequence is:

```bash
python3 "$SKILL_ROOT/scripts/overlay_setup.py" add-capture \
  --target "$TARGET_CALLABLE" --out "$CAPTURE_DIR" \
  --overlay "$CAPTURE_OVERLAY" --from "$BASELINE_OVERLAY" \
  --capture-file "$SKILL_ROOT/scripts/capture_shapes.py"
python3 "$SKILL_ROOT/scripts/overlay_setup.py" add-marker \
  --target "$TARGET_CALLABLE" --overlay "$CAPTURE_OVERLAY" \
  --marker-file "$SKILL_ROOT/scripts/seam_trace.py"
```

Pass the capture overlay through the serving process's `PYTHONPATH`; preserve
the existing baseline module/search paths and environment. Mark all relevant
candidate callsites, including cached aliases, with separate `add-marker`
calls as appropriate. Native/JIT kernel objects may not be safely monkeypatched;
trace their real Python launchers and native launches instead.

Read the helpers' environment controls before choosing budgets and profiler
windows. Do not assume their defaults capture every relevant shape or phase.
`capture_shapes.py` handles tensors and ordinary containers; opaque `self`, KV
getters, routing/state or multiple phase targets may require a task-local
serializer. Follow [source extraction](source-extraction.md) for byte accounting,
shared-state, independent phase state and replay checks. Keep customizations
task-local, with source hashes and a bounded small replay before a larger sweep.

Once captured, verify target-call markers, host-launch correlation, real device
kernel names, grids, invocation counts, phase identity and rank integrity using
`kernel_selection.py`. Pass `--required-device-kernel` only for simultaneous
mandatory members, not alternative dispatch. Run `--help` for its input contract.
Store its complete JSON, raw trace and metadata. The CLI may promote/reclaim
oracles under `--task-dir`; use a separate directory per phase and preserve
needed captures before allowing that cleanup. No evidence from one phase may
overwrite another phase's oracle.

Replay one captured baseline case per declared phase before expanding source
extraction. Successful file persistence alone does not prove complete inputs,
correct state reconstruction or faithful whole-call timing.
