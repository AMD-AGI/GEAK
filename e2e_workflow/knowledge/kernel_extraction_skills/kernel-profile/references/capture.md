# Bounded profile evidence capture

Capture belongs to profile. UT authoring consumes the handoff, generates numerical
inputs from seeds and asks profile for missing evidence when necessary. No capture
persists tensor payloads or input/output oracles.

Plan points with [workload measurement](dynamic-workloads.md) from the actual serving
configuration. Reuse compatible traces before taking another capture. There is no
universal server-start command in this skill: read the user's actual serving launch and
request-client commands, and take flags, model configuration, health API, tokenizer and
client APIs from the installed backend. `serving_probe.py` drives one planned point
against that server: steady state, one armed window, explicit close, trace verification.

Record request count, concurrency, token/context controls, timeout and capture
trigger/length before launch. Inspect wrappers and warmup hooks: zero measured repeats
need not mean zero client requests. Disable any outer full-replay warmup for this bounded
probe and verify the effective client. Probe measurements are diagnostic and do not
replace baseline throughput or the original workload distribution.

Preserve serving TP and the allocated physical devices. Run an owned serving probe under
a full-set lease, with the launcher in the foreground for the entire server/client
lifecycle. It may start child processes only if it records their PIDs/process group,
traps exit/signals, stops and waits for those owned children before returning. Apply
bounded health polling; stop polling if the server exits. Do not attach cleanup to
unrelated processes or stop an existing service unless the user authorized that
lifecycle.

When live callable attribution needs a bounded diagnostic trace, mark the real Python
launchers with an isolated overlay on top of the accepted baseline:

```bash
python3 "$SHARED_TOOLS/overlay_setup.py" add-marker \
  --target "$TARGET_CALLABLE" --overlay "$CAPTURE_OVERLAY" \
  --from "$BASELINE_OVERLAY" --marker-file "$SHARED_TOOLS/seam_trace.py"
```

Apply its `PYTHONPATH` only to the owned diagnostic process and keep the accepted
baseline's other import settings. Mark every candidate callsite, including cached
aliases. Native/JIT objects may require marking their real Python launcher instead.

Use `kernel_selection.py --strict-launch` with the process metadata and raw traces to
verify target-call markers, host-launch correlation, real device kernel names, grids,
invocation counts, phase identity and rank integrity. Pass `--required-device-kernel`
only for simultaneous mandatory members, not alternative dispatch; run `--help` for its
input contract. Store its complete JSON, raw trace and metadata. Do not pass `--task-dir`:
there are no captured oracles to promote.

Shapes, dtypes, strides, quantization and call counts are recorded as metadata in the
ranking and handoff. No capture persists tensors: there are no input/output oracles and no
structural fixtures. Where routing, page tables or sequence state cannot be reconstructed
from metadata, record the gap in the handoff; the UT reports it as a blocker or a declared
approximation.
