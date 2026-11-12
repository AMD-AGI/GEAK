# Bounded profile evidence capture

Capture belongs to profile. UT authoring consumes the handoff, generates numerical
inputs from seeds and requests missing evidence from profile when necessary.

Use [shared sampling](multi-config-sampling.md) with the actual serving configuration
and pinned runtime identity. Reuse compatible traces before taking another capture.
`probe_server.py` owns the serving lifecycle; `run_points.py` captures planned points,
checks phase/shape, drains requests and produces the per-rank census/source evidence.
`compile_watch.py` records compile variants and call descriptors, retaining necessary
structural values while describing bulk operands without saving their full payloads.

Preserve serving TP and the allocated physical devices. Hold a full-set GPU lease for
the foreground server/client lifecycle. Record owned PIDs, stop and wait for workers
on exit, and use bounded health polling. Do not clean up unrelated services. Record
request count, concurrency, context, capture trigger/length and graph/eager mode.
Diagnostic probe costs do not replace the original workload distribution or baseline
throughput.

When live callable attribution needs a bounded diagnostic trace, mark the real
Python launchers using the existing overlay:

```bash
python3 "$SHARED_TOOLS/overlay_setup.py" add-marker \
  --target "$TARGET_CALLABLE" --overlay "$CAPTURE_OVERLAY" \
  --from "$BASELINE_OVERLAY" --marker-file "$SHARED_TOOLS/seam_trace.py"
```

Apply its `PYTHONPATH` only to the owned diagnostic process and retain the accepted
baseline's other import settings. Mark every candidate callsite, including cached
aliases. Native/JIT objects may require marking their real Python launcher instead.

Use `kernel_selection.py --strict-launch` with the process metadata and raw traces.
Pass `--required-device-kernel` only for simultaneous mandatory members. Preserve the
full verdict, grids, invocation counts, phase and rank evidence. The selector reads
these inputs without promoting or deleting captured tensor files.

The old persistent input/output oracle capture and regeneration path is retired.
New tasks follow [fixed-contract authoring](task-authoring.md). Keep minimal structural
fixtures only where descriptors/reconstruction cannot preserve the required semantics.
